package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.LinearWeightInitialization;
import io.github.pho001.synaptik.nn.layers.Linear;
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
import java.util.stream.Collectors;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import org.junit.jupiter.api.Test;

class StateDictionaryTest {
    @Test
    void exposesOnlyTheExactFinalStateTypesAndModuleMethods() throws ReflectiveOperationException {
        Method export = Module.class.getDeclaredMethod("stateDictionary");
        Method load = Module.class.getDeclaredMethod("loadStateDictionary", StateDictionary.class);

        assertAll(
                () -> assertTrue(Modifier.isPublic(StateKind.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(StateKind.class.getModifiers())),
                () -> assertTrue(StateKind.class.isEnum()),
                () -> assertArrayEquals(
                        new StateKind[] {StateKind.PARAMETER, StateKind.BUFFER}, StateKind.values()),
                () -> assertTrue(Modifier.isPublic(StateEntry.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(StateEntry.class.getModifiers())),
                () -> assertTrue(StateEntry.class.isRecord()),
                () -> assertEquals(
                        List.of(String.class, StateKind.class, Tensor.class),
                        Arrays.stream(StateEntry.class.getRecordComponents())
                                .map(component -> component.getType())
                                .toList()),
                () -> assertTrue(Modifier.isPublic(StateDictionary.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(StateDictionary.class.getModifiers())),
                () -> assertTrue(StateDictionary.class.isRecord()),
                () -> assertEquals(
                        List.of(List.class),
                        Arrays.stream(StateDictionary.class.getRecordComponents())
                                .map(component -> component.getType())
                                .toList()),
                () -> assertTrue(Modifier.isPublic(export.getModifiers())),
                () -> assertTrue(Modifier.isFinal(export.getModifiers())),
                () -> assertEquals(StateDictionary.class, export.getReturnType()),
                () -> assertTrue(Modifier.isPublic(load.getModifiers())),
                () -> assertTrue(Modifier.isFinal(load.getModifiers())),
                () -> assertEquals(void.class, load.getReturnType()),
                () -> assertEquals(
                        Set.of("stateDictionary", "loadStateDictionary"),
                        Arrays.stream(Module.class.getDeclaredMethods())
                                .filter(method -> Modifier.isPublic(method.getModifiers()))
                                .map(Method::getName)
                                .filter(name -> name.toLowerCase().contains("state"))
                                .collect(Collectors.toSet())));
    }

    @Test
    void stateEntryValidatesComponentsInOrderAndPreservesAcceptedPathAndTensorIdentity() {
        Tensor value = parameterTensor(DataType.FLOAT64, Shape.of(2), "value");
        String accepted = " encoder . weight ";
        StateEntry entry = new StateEntry(accepted, StateKind.PARAMETER, value);

        assertEquals(accepted, entry.path());
        assertSame(StateKind.PARAMETER, entry.kind());
        assertSame(value, entry.value());
        assertEquals("path", assertThrows(
                NullPointerException.class,
                () -> new StateEntry(null, null, null)).getMessage());
        assertTrue(assertThrows(
                        IllegalArgumentException.class,
                        () -> new StateEntry(".", null, null))
                .getMessage().contains("path"));
        assertEquals("kind", assertThrows(
                NullPointerException.class,
                () -> new StateEntry("weight", null, null)).getMessage());
        assertEquals("value", assertThrows(
                NullPointerException.class,
                () -> new StateEntry("weight", StateKind.PARAMETER, null)).getMessage());

        for (String invalid : List.of("", " ", ".weight", "weight.", "child..weight", "child. .weight")) {
            assertThrows(
                    IllegalArgumentException.class,
                    () -> new StateEntry(invalid, StateKind.PARAMETER, value),
                    invalid);
        }
    }

    @Test
    void dictionaryDefensivelyCopiesInOrderAndRejectsNullAndFirstDuplicatePath() {
        StateEntry weight = entry("weight", StateKind.PARAMETER, parameterTensor());
        StateEntry mean = entry("runningMean", StateKind.BUFFER, bufferTensor());
        List<StateEntry> supplied = new ArrayList<>(List.of(mean, weight));
        StateDictionary dictionary = new StateDictionary(supplied);

        supplied.clear();
        assertEquals(List.of(mean, weight), dictionary.entries());
        assertThrows(UnsupportedOperationException.class, () -> dictionary.entries().clear());
        assertEquals("entries", assertThrows(
                NullPointerException.class, () -> new StateDictionary(null)).getMessage());
        assertEquals("entries[1]", assertThrows(
                NullPointerException.class,
                () -> new StateDictionary(Arrays.asList(weight, null))).getMessage());
        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class,
                () -> new StateDictionary(List.of(
                        weight,
                        mean,
                        entry("weight", StateKind.BUFFER, bufferTensor()))));
        assertTrue(duplicate.getMessage().contains("weight"));
    }

    @Test
    void exportUsesCombinedDeterministicDepthFirstOrderingAndExactReferences() {
        StateModule root = new StateModule("rootWeight", "rootBuffer", 1.0d);
        StateModule first = new StateModule("weight", "mean", 2.0d);
        StateModule nested = new StateModule("scale", "variance", 3.0d);
        StateModule second = new StateModule("table", "counter", 4.0d);
        first.attach("nested", nested);
        root.attach("first", first);
        root.attach("second", second);

        StateDictionary dictionary = root.stateDictionary();

        assertEquals(
                List.of(
                        "rootWeight", "rootBuffer",
                        "first.weight", "first.mean",
                        "first.nested.scale", "first.nested.variance",
                        "second.table", "second.counter"),
                dictionary.entries().stream().map(StateEntry::path).toList());
        assertEquals(
                List.of(
                        StateKind.PARAMETER, StateKind.BUFFER,
                        StateKind.PARAMETER, StateKind.BUFFER,
                        StateKind.PARAMETER, StateKind.BUFFER,
                        StateKind.PARAMETER, StateKind.BUFFER),
                dictionary.entries().stream().map(StateEntry::kind).toList());
        assertSame(root.parameter.value(), dictionary.entries().get(0).value());
        assertSame(root.buffer.value(), dictionary.entries().get(1).value());
        assertSame(nested.parameter.value(), dictionary.entries().get(4).value());
        assertSame(second.buffer.value(), dictionary.entries().get(7).value());
    }

    @Test
    void exportIsAValueSnapshotWhileDiscoverySnapshotsRetainLiveWrappers() {
        StateModule module = new StateModule("weight", "mean", 1.0d);
        StateDictionary before = module.stateDictionary();
        Map<String, Parameter> parameterDiscovery = module.parametersRecursively();
        Map<String, Buffer> bufferDiscovery = module.buffersRecursively();
        Tensor oldParameter = module.parameter.value();
        Tensor oldBuffer = module.buffer.value();
        Tensor oldExpression = oldParameter.add(parameterTensor());
        Tensor nextParameter = parameterTensor(DataType.FLOAT64, Shape.scalar(), "next");
        Tensor nextBuffer = bufferTensor(DataType.FLOAT64, Shape.scalar(), true, "next-buffer");

        module.parameter.replace(nextParameter);
        module.replaceDirectBuffer("mean", nextBuffer);

        assertSame(oldParameter, before.entries().get(0).value());
        assertSame(oldBuffer, before.entries().get(1).value());
        assertSame(nextParameter, parameterDiscovery.get("weight").value());
        assertSame(nextBuffer, bufferDiscovery.get("mean").value());
        assertSame(oldParameter, oldExpression.provenance().orElseThrow().inputs().getFirst());
    }

    @Test
    void loadAcceptsReorderedEntriesAndPreservesWrappersTreeModeAndExactCandidateValues() {
        StateModule root = new StateModule("weight", "mean", 1.0d);
        StateModule child = new StateModule("scale", "variance", 2.0d);
        root.attach("child", child);
        root.eval();
        Parameter rootParameter = root.parameter;
        Buffer rootBuffer = root.buffer;
        Parameter childParameter = child.parameter;
        Buffer childBuffer = child.buffer;
        Map<String, Module> children = root.children();
        Map<String, Parameter> parameters = root.parametersRecursively();
        Map<String, Buffer> buffers = root.buffersRecursively();
        Tensor nextRootParameter = parameterTensor(DataType.FLOAT64, Shape.scalar(), "rp");
        Tensor nextRootBuffer = bufferTensor(DataType.FLOAT64, Shape.scalar(), true, "rb");
        Tensor nextChildParameter = parameterTensor(DataType.FLOAT64, Shape.scalar(), "cp");
        Tensor nextChildBuffer = bufferTensor(DataType.FLOAT64, Shape.scalar(), false, "cb");
        StateDictionary candidate = new StateDictionary(List.of(
                entry("child.variance", StateKind.BUFFER, nextChildBuffer),
                entry("weight", StateKind.PARAMETER, nextRootParameter),
                entry("child.scale", StateKind.PARAMETER, nextChildParameter),
                entry("mean", StateKind.BUFFER, nextRootBuffer)));

        root.loadStateDictionary(candidate);

        assertAll(
                () -> assertSame(nextRootParameter, rootParameter.value()),
                () -> assertSame(nextRootBuffer, rootBuffer.value()),
                () -> assertSame(nextChildParameter, childParameter.value()),
                () -> assertSame(nextChildBuffer, childBuffer.value()),
                () -> assertSame(rootParameter, parameters.get("weight")),
                () -> assertSame(rootBuffer, buffers.get("mean")),
                () -> assertSame(childParameter, parameters.get("child.scale")),
                () -> assertSame(childBuffer, buffers.get("child.variance")),
                () -> assertSame(child, children.get("child")),
                () -> assertEquals(ForwardMode.EVALUATION, root.mode()),
                () -> assertEquals(ForwardMode.EVALUATION, child.mode()));
    }

    @Test
    void loadRejectsNullThenMissingBeforeUnexpectedWithoutMutation() {
        StateModule module = new StateModule("weight", "mean", 1.0d);
        Tensor oldParameter = module.parameter.value();
        Tensor oldBuffer = module.buffer.value();

        assertEquals("dictionary", assertThrows(
                NullPointerException.class, () -> module.loadStateDictionary(null)).getMessage());
        StateDictionary candidate = new StateDictionary(List.of(
                entry("mean", StateKind.BUFFER, bufferTensor()),
                entry("extra", StateKind.BUFFER, bufferTensor())));
        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class, () -> module.loadStateDictionary(candidate));

        assertTrue(failure.getMessage().contains("missing"));
        assertTrue(failure.getMessage().contains("weight"));
        assertSame(oldParameter, module.parameter.value());
        assertSame(oldBuffer, module.buffer.value());
    }

    @Test
    void loadRejectsUnexpectedCandidateInCandidateEncounterOrderWithoutMutation() {
        EmptyModule module = new EmptyModule();
        StateDictionary candidate = new StateDictionary(List.of(
                entry("second", StateKind.BUFFER, bufferTensor()),
                entry("first", StateKind.BUFFER, bufferTensor())));

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class, () -> module.loadStateDictionary(candidate));

        assertTrue(failure.getMessage().contains("unexpected"));
        assertTrue(failure.getMessage().contains("second"));
    }

    @Test
    void loadValidatesKindThenDataTypeThenShapeThenParameterGradient() {
        StateModule module = new StateModule("weight", "mean", 1.0d);
        Tensor oldParameter = module.parameter.value();
        Tensor oldBuffer = module.buffer.value();
        Tensor validBuffer = bufferTensor();

        IllegalArgumentException kind = assertThrows(
                IllegalArgumentException.class,
                () -> module.loadStateDictionary(dictionary(
                        entry("weight", StateKind.BUFFER,
                                bufferTensor(DataType.FLOAT32, Shape.of(3), false, "all-wrong")),
                        entry("mean", StateKind.BUFFER, validBuffer))));
        IllegalArgumentException type = assertThrows(
                IllegalArgumentException.class,
                () -> module.loadStateDictionary(dictionary(
                        entry("weight", StateKind.PARAMETER,
                                bufferTensor(DataType.FLOAT32, Shape.of(3), false, "wrong-type")),
                        entry("mean", StateKind.BUFFER, validBuffer))));
        IllegalArgumentException shape = assertThrows(
                IllegalArgumentException.class,
                () -> module.loadStateDictionary(dictionary(
                        entry("weight", StateKind.PARAMETER,
                                bufferTensor(DataType.FLOAT64, Shape.of(3), false, "wrong-shape")),
                        entry("mean", StateKind.BUFFER, validBuffer))));
        IllegalArgumentException gradient = assertThrows(
                IllegalArgumentException.class,
                () -> module.loadStateDictionary(dictionary(
                        entry("weight", StateKind.PARAMETER,
                                bufferTensor(DataType.FLOAT64, Shape.scalar(), false, "no-gradient")),
                        entry("mean", StateKind.BUFFER, validBuffer))));

        assertAll(
                () -> assertTrue(kind.getMessage().contains("kind")),
                () -> assertTrue(type.getMessage().contains("data type")),
                () -> assertTrue(shape.getMessage().contains("shape")),
                () -> assertTrue(gradient.getMessage().contains("requiresGrad")),
                () -> assertSame(oldParameter, module.parameter.value()),
                () -> assertSame(oldBuffer, module.buffer.value()));
    }

    @Test
    void bufferKindMismatchFailsBeforeAnyBindingChanges() {
        StateModule module = new StateModule("weight", "mean", 1.0d);
        Tensor oldParameter = module.parameter.value();
        Tensor oldBuffer = module.buffer.value();
        StateDictionary candidate = dictionary(
                entry("weight", StateKind.PARAMETER, parameterTensor()),
                entry("mean", StateKind.PARAMETER, parameterTensor()));

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class, () -> module.loadStateDictionary(candidate));

        assertTrue(failure.getMessage().contains("kind"));
        assertTrue(failure.getMessage().contains("mean"));
        assertSame(oldParameter, module.parameter.value());
        assertSame(oldBuffer, module.buffer.value());
    }

    @Test
    void laterCompatibilityFailureLeavesEveryEarlierAndLaterBindingUnchanged() {
        StateModule root = new StateModule("weight", "mean", 1.0d);
        StateModule child = new StateModule("scale", "variance", 2.0d);
        root.attach("child", child);
        List<Tensor> old = root.stateDictionary().entries().stream().map(StateEntry::value).toList();
        StateDictionary candidate = dictionary(
                entry("weight", StateKind.PARAMETER, parameterTensor()),
                entry("mean", StateKind.BUFFER, bufferTensor(DataType.FLOAT64, Shape.scalar(), true, "mean")),
                entry("child.scale", StateKind.PARAMETER, parameterTensor()),
                entry("child.variance", StateKind.BUFFER,
                        bufferTensor(DataType.FLOAT64, Shape.of(2), false, "wrong")));

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class, () -> root.loadStateDictionary(candidate));

        assertTrue(failure.getMessage().contains("child.variance"));
        assertEquals(old, root.stateDictionary().entries().stream().map(StateEntry::value).toList());
        for (int index = 0; index < old.size(); index++) {
            assertSame(old.get(index), root.stateDictionary().entries().get(index).value());
        }
    }

    @Test
    void compatibilityIgnoresIdentityLabelLayoutStorageProvenanceAndBufferGradient() {
        Shape shape = Shape.of(2);
        Tensor originalParameter = parameterTensor(DataType.FLOAT64, shape, "original");
        Tensor originalBuffer = bufferTensor(DataType.FLOAT64, shape, false, "original-buffer");
        SuppliedModule module = new SuppliedModule(originalParameter, originalBuffer);
        Tensor parameterLeaf = parameterTensor(DataType.FLOAT64, shape, "candidate-input");
        Tensor parameterExpression = parameterLeaf.add(parameterTensor(
                DataType.FLOAT64, shape, "candidate-right"));
        Tensor bufferInput = parameterTensor(DataType.FLOAT64, shape, "buffer-input");
        Tensor bufferExpression = bufferInput.neg();

        assertTrue(originalParameter.hostStorage().isPresent());
        assertTrue(parameterExpression.hostStorage().isEmpty());
        assertTrue(parameterExpression.descriptor().layout().isEmpty());
        assertTrue(parameterExpression.provenance().isPresent());
        assertTrue(bufferExpression.descriptor().requiresGrad());
        assertNotSame(originalParameter, parameterExpression);
        module.loadStateDictionary(dictionary(
                entry("weight", StateKind.PARAMETER, parameterExpression),
                entry("mean", StateKind.BUFFER, bufferExpression)));

        assertSame(parameterExpression, module.parameter.value());
        assertSame(bufferExpression, module.buffer.value());
    }

    @Test
    void emptyModulesRoundTripAndRejectNonEmptyState() {
        EmptyModule module = new EmptyModule();
        StateDictionary empty = module.stateDictionary();

        assertTrue(empty.entries().isEmpty());
        module.loadStateDictionary(new StateDictionary(List.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> module.loadStateDictionary(dictionary(
                        entry("unexpected", StateKind.BUFFER, bufferTensor()))));
    }

    @Test
    void deepTreeExportAndLoadUseExplicitStack() {
        int depth = 5_000;
        StateModule leaf = new StateModule("weight", "mean", 1.0d);
        Module root = leaf;
        for (int index = 0; index < depth; index++) {
            EmptyModule parent = new EmptyModule();
            parent.attach("next", root);
            root = parent;
        }
        String prefix = "next.".repeat(depth);
        StateDictionary exported = root.stateDictionary();
        Tensor nextParameter = parameterTensor();
        Tensor nextBuffer = bufferTensor(DataType.FLOAT64, Shape.scalar(), true, "next");

        assertEquals(List.of(prefix + "weight", prefix + "mean"),
                exported.entries().stream().map(StateEntry::path).toList());
        root.loadStateDictionary(dictionary(
                entry(prefix + "mean", StateKind.BUFFER, nextBuffer),
                entry(prefix + "weight", StateKind.PARAMETER, nextParameter)));
        assertSame(nextParameter, leaf.parameter.value());
        assertSame(nextBuffer, leaf.buffer.value());
    }

    @Test
    void malformedRepeatedIdentityFailsBeforeExportOrAnyLoadInstallation()
            throws ReflectiveOperationException {
        StateModule root = new StateModule("weight", "mean", 1.0d);
        StateModule child = new StateModule("scale", "variance", 2.0d);
        root.attach("child", child);
        StateDictionary candidate = root.stateDictionary();
        List<Tensor> old = candidate.entries().stream().map(StateEntry::value).toList();
        corruptChildren(child).put("cycle", root);

        assertThrows(IllegalStateException.class, root::stateDictionary);
        assertThrows(IllegalStateException.class, () -> root.loadStateDictionary(candidate));
        assertSame(old.get(0), root.parameter.value());
        assertSame(old.get(1), root.buffer.value());
        assertSame(old.get(2), child.parameter.value());
        assertSame(old.get(3), child.buffer.value());
    }

    @Test
    void strictLoadInitializesAutomaticLinearFromExactCandidateReferencesWithoutForward()
            throws ReflectiveOperationException {
        Linear layer = automaticLinear(true);
        Tensor weight = parameterTensor(DataType.FLOAT32, Shape.of(4, 3), "loaded-weight");
        Tensor bias = parameterTensor(DataType.FLOAT32, Shape.of(4), "loaded-bias");
        AtomicLong next = nextTensorIdState();
        long beforeLoad = next.get();

        layer.loadStateDictionary(dictionary(
                entry("bias", StateKind.PARAMETER, bias),
                entry("weight", StateKind.PARAMETER, weight)));

        assertAll(
                () -> assertSame(weight, layer.weight().value()),
                () -> assertSame(bias, layer.bias().orElseThrow().value()),
                () -> assertEquals(List.of("weight", "bias"), layer.stateDictionary().entries()
                        .stream().map(StateEntry::path).toList()),
                () -> assertSame(weight, layer.stateDictionary().entries().get(0).value()),
                () -> assertSame(bias, layer.stateDictionary().entries().get(1).value()),
                () -> assertEquals(beforeLoad, next.get()));
    }

    @Test
    void strictLoadValidatesWholeTreeBeforeReplacingExistingOrPublishingReservations() {
        StateModule existing = new StateModule("weight", "mean", 1.0d);
        Tensor oldWeight = existing.parameter.value();
        Tensor oldBuffer = existing.buffer.value();
        Linear automatic = automaticLinear(true);
        EmptyModule root = new EmptyModule();
        root.attach("existing", existing);
        root.attach("automatic", automatic);
        Tensor nextWeight = parameterTensor();
        Tensor nextBuffer = bufferTensor();
        Tensor candidateWeight = parameterTensor(
                DataType.FLOAT32, Shape.of(4, 3), "candidate-weight");
        Tensor wrongBias = parameterTensor(
                DataType.FLOAT32, Shape.of(5), "wrong-bias");

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> root.loadStateDictionary(dictionary(
                        entry("existing.weight", StateKind.PARAMETER, nextWeight),
                        entry("existing.mean", StateKind.BUFFER, nextBuffer),
                        entry("automatic.weight", StateKind.PARAMETER, candidateWeight),
                        entry("automatic.bias", StateKind.PARAMETER, wrongBias))));

        assertAll(
                () -> assertTrue(failure.getMessage().contains("automatic.bias")),
                () -> assertSame(oldWeight, existing.parameter.value()),
                () -> assertSame(oldBuffer, existing.buffer.value()),
                () -> assertThrows(IllegalStateException.class, automatic::weight),
                () -> assertThrows(IllegalStateException.class, root::stateDictionary));
    }

    @Test
    void reservedStrictLoadValidatesKindThenConfiguredTypeThenShapeThenGradient() {
        Linear kindLayer = automaticLinear(false);
        Linear typeLayer = automaticLinear(false);
        Linear shapeLayer = automaticLinear(false);
        Linear gradientLayer = automaticLinear(false);

        IllegalArgumentException kind = assertThrows(
                IllegalArgumentException.class,
                () -> kindLayer.loadStateDictionary(dictionary(entry(
                        "weight",
                        StateKind.BUFFER,
                        bufferTensor(DataType.FLOAT64, Shape.of(5), false, "all-wrong")))));
        IllegalArgumentException type = assertThrows(
                IllegalArgumentException.class,
                () -> typeLayer.loadStateDictionary(dictionary(entry(
                        "weight",
                        StateKind.PARAMETER,
                        bufferTensor(DataType.FLOAT64, Shape.of(5), false, "wrong-type")))));
        IllegalArgumentException shape = assertThrows(
                IllegalArgumentException.class,
                () -> shapeLayer.loadStateDictionary(dictionary(entry(
                        "weight",
                        StateKind.PARAMETER,
                        bufferTensor(DataType.FLOAT32, Shape.of(5), false, "wrong-shape")))));
        IllegalArgumentException gradient = assertThrows(
                IllegalArgumentException.class,
                () -> gradientLayer.loadStateDictionary(dictionary(entry(
                        "weight",
                        StateKind.PARAMETER,
                        bufferTensor(DataType.FLOAT32, Shape.of(4, 3), false, "no-gradient")))));

        assertAll(
                () -> assertTrue(kind.getMessage().contains("kind")),
                () -> assertTrue(type.getMessage().contains("data type")),
                () -> assertTrue(shape.getMessage().contains("rank two")),
                () -> assertTrue(gradient.getMessage().contains("requiresGrad")),
                () -> assertThrows(IllegalStateException.class, kindLayer::weight),
                () -> assertThrows(IllegalStateException.class, typeLayer::weight),
                () -> assertThrows(IllegalStateException.class, shapeLayer::weight),
                () -> assertThrows(IllegalStateException.class, gradientLayer::weight));
    }

    @Test
    void eagerAndInitializedAutomaticLinearDictionariesLoadInBothDirections() {
        Linear eager = new Linear(
                parameterTensor(DataType.FLOAT32, Shape.of(4, 3), "eager-weight"),
                parameterTensor(DataType.FLOAT32, Shape.of(4), "eager-bias"));
        Linear automatic = automaticLinear(true);

        automatic.loadStateDictionary(eager.stateDictionary());
        Tensor automaticWeight = automatic.weight().value();
        Tensor automaticBias = automatic.bias().orElseThrow().value();
        Linear eagerTarget = new Linear(
                parameterTensor(DataType.FLOAT32, Shape.of(4, 3), "target-weight"),
                parameterTensor(DataType.FLOAT32, Shape.of(4), "target-bias"));
        eagerTarget.loadStateDictionary(automatic.stateDictionary());

        assertAll(
                () -> assertSame(eager.weight().value(), automaticWeight),
                () -> assertSame(eager.bias().orElseThrow().value(), automaticBias),
                () -> assertSame(automaticWeight, eagerTarget.weight().value()),
                () -> assertSame(automaticBias, eagerTarget.bias().orElseThrow().value()));
    }

    private static Linear automaticLinear(boolean bias) {
        RandomGeneratorFactory<RandomGenerator> factory =
                RandomGeneratorFactory.of("L64X128MixRandom");
        return new Linear(
                4,
                bias,
                DataType.FLOAT32,
                LinearWeightInitialization.GLOROT_UNIFORM,
                factory,
                7L);
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static StateDictionary dictionary(StateEntry... entries) {
        return new StateDictionary(List.of(entries));
    }

    private static StateEntry entry(String path, StateKind kind, Tensor value) {
        return new StateEntry(path, kind, value);
    }

    private static Tensor parameterTensor() {
        return parameterTensor(DataType.FLOAT64, Shape.scalar(), "parameter");
    }

    private static Tensor parameterTensor(DataType type, Shape shape, String label) {
        return TensorFactory.zeros(shape, type, Optional.of(label), true);
    }

    private static Tensor bufferTensor() {
        return bufferTensor(DataType.FLOAT64, Shape.scalar(), false, "buffer");
    }

    private static Tensor bufferTensor(
            DataType type, Shape shape, boolean requiresGrad, String label) {
        return TensorFactory.zeros(shape, type, Optional.of(label), requiresGrad);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Module> corruptChildren(Module module)
            throws ReflectiveOperationException {
        Field children = Module.class.getDeclaredField("children");
        children.setAccessible(true);
        return (Map<String, Module>) children.get(module);
    }

    private static class EmptyModule extends Module {
        final <T extends Module> T attach(String name, T child) {
            return child(name, child);
        }
    }

    private static final class StateModule extends EmptyModule {
        private final Parameter parameter;
        private final Buffer buffer;

        private StateModule(String parameterName, String bufferName, double value) {
            parameter = parameter(parameterName, TensorFactory.scalar(value, Optional.empty(), true));
            buffer = buffer(bufferName, TensorFactory.scalar(value, Optional.empty(), false));
        }

        private void replaceDirectBuffer(String name, Tensor value) {
            replaceBuffer(name, value);
        }
    }

    private static final class SuppliedModule extends Module {
        private final Parameter parameter;
        private final Buffer buffer;

        private SuppliedModule(Tensor parameter, Tensor buffer) {
            this.parameter = parameter("weight", parameter);
            this.buffer = buffer("mean", buffer);
        }
    }
}
