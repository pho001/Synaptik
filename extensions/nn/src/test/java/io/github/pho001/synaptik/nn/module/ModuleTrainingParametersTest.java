package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.training.AdamW;
import io.github.pho001.synaptik.training.GradientMode;
import io.github.pho001.synaptik.training.Sgd;
import io.github.pho001.synaptik.training.TrainingParameterStorage;
import io.github.pho001.synaptik.training.TrainingParameters;
import io.github.pho001.synaptik.training.TrainingSession;
import io.github.pho001.synaptik.training.TrainingState;
import java.lang.foreign.Arena;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Field;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.random.RandomGeneratorFactory;
import org.junit.jupiter.api.Test;

/** Real NN ownership/traversal remains behind the generic live admission adapter. */
final class ModuleTrainingParametersTest {
    @Test
    void snapshotsRootFirstDepthFirstOrderAndIgnoresBuffersModesAndBytes() {
        Node root = new Node();
        Tensor z = heap(2f);
        root.add("z", z);
        Node child = root.attach("encoder", new Node());
        Tensor nested = heap(3f);
        child.add("a", nested);
        root.add("b", heap(4f));
        TrainingParameters source = root.trainingParameters();
        assertEquals(List.of("z", "b", "encoder.a"),
                source.entries().stream().map(TrainingParameters.Entry::path).toList());
        assertSame(z, source.entries().getFirst().tensor());
        assertSame(nested, source.entries().getLast().tensor());
        assertThrows(UnsupportedOperationException.class, () -> source.entries().clear());
        root.buffer("running", heap(0f));
        root.eval();
        root.replaceBuffer("running", heap(9f));
        z.hostStorage().orElseThrow().segment().set(ValueLayout.JAVA_FLOAT_UNALIGNED, 0, 7f);
        source.validateUnchanged();
        child.add("later", heap(5f));
        assertEquals(3, source.entries().size());
        assertThrows(IllegalStateException.class, source::validateUnchanged);
    }

    @Test
    void countOrderPathAndWrapperChangesRejectEvenWhenFrozenTensorEntriesLookUnchanged()
            throws ReflectiveOperationException {
        for (String change : List.of("count", "order", "path", "wrapper")) {
            Node root = new Node();
            Tensor first = heap(2f);
            root.add("first", first);
            root.add("second", heap(3f));
            TrainingParameters source = root.trainingParameters();
            Map<String, Object> declarations = declarations(root);
            switch (change) {
                case "count" -> root.add("third", heap(4f));
                case "order" -> {
                    Object slot = declarations.remove("first");
                    declarations.put("first", slot);
                }
                case "path" -> {
                    Object slot = declarations.remove("second");
                    declarations.put("renamed", slot);
                }
                case "wrapper" -> {
                    Object slot = declarations.get("first");
                    Field wrapper = slot.getClass().getDeclaredField("parameter");
                    wrapper.setAccessible(true);
                    wrapper.set(slot, new Parameter("first", first));
                    assertSame(first, root.parametersRecursively().get("first").value());
                }
                default -> throw new AssertionError(change);
            }
            assertEquals(List.of("first", "second"),
                    source.entries().stream().map(TrainingParameters.Entry::path).toList());
            assertThrows(IllegalStateException.class, source::validateUnchanged, change);
        }
    }

    @Test
    void directReplacementAndStateLoadRejectAgainstTheLiveRootNotFrozenEntries() {
        for (boolean load : List.of(false, true)) {
            Node root = new Node();
            Node child = root.attach("child", new Node());
            Tensor original = heap(2f);
            Parameter wrapper = child.add("weight", original);
            TrainingParameters source = root.trainingParameters();
            Tensor replacement = heap(3f);
            if (load) {
                root.loadStateDictionary(new StateDictionary(List.of(
                        new StateEntry("child.weight", StateKind.PARAMETER, replacement))));
            } else {
                wrapper.replace(replacement);
            }
            assertSame(wrapper, child.parameters().getFirst());
            assertSame(original, source.entries().getFirst().tensor());
            assertThrows(IllegalStateException.class, source::validateUnchanged);
            wrapper.replace(original);
            source.validateUnchanged();
        }
    }

    @Test
    void emptyAndDeferredDiscoveryRetainFailureBeforeNativeCopy() {
        Linear deferred = new Linear(1, true, DataType.FLOAT32,
                ParameterInitialization.glorotUniform(),
                RandomGeneratorFactory.of("L64X128MixRandom"), 41L);
        Node root = new Node();
        root.add("first", heap(1f));
        root.attach("deferred", deferred);
        assertThrows(IllegalStateException.class, root::trainingParameters);
        assertThrows(IllegalStateException.class, deferred::trainingParameters);
        try (Arena arena = Arena.ofShared()) {
            assertThrows(IllegalStateException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(new Node().trainingParameters(), arena));
            assertThrows(IllegalStateException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(deferred.trainingParameters(), arena));
        }
        // Finalizing deferred state allows a complete live borrow; no initializer is bypassed.
        deferred.forward(TensorFactory.scalar(1f, Optional.empty(), false).reshape(1));
        root.trainingParameters().validateUnchanged();
    }

    @Test
    void sameAdapterSurvivesNativeCopyAndOpensSessionWithoutReplacingWrappersOrTensors() {
        Node root = new Node();
        Tensor p = heap(2f);
        Parameter wrapper = root.add("weight", p);
        TrainingParameters source = root.trainingParameters();
        var oldStorage = p.hostStorage().orElseThrow();
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            TrainingParameterStorage.copyToSharedNative(source, arena);
            source.validateUnchanged();
            assertSame(wrapper, root.parameters().getFirst());
            assertSame(p, wrapper.value());
            assertSame(p, source.entries().getFirst().tensor());
            assertNotSame(oldStorage, p.hostStorage().orElseThrow());
            try (TrainingSession session = TrainingSession.open(engine, source, p.mul(p), new Sgd(0.1))) {
                session.run(List.of(), GradientMode.RESET_AND_STEP);
                assertEquals(1.6f, p.hostStorage().orElseThrow().segment().get(ValueLayout.JAVA_FLOAT, 0), 1e-6f);
                assertEquals(1, session.optimizerSteps());
            }
            assertTrue(arena.scope().isAlive());
            assertTrue(oldStorage.isAlive());
            assertEquals(2f, oldStorage.segment().get(ValueLayout.JAVA_FLOAT_UNALIGNED, 0));
        }
    }

    @Test
    void aliasedNnPathsAreNotSplitByCopyOrAdmission() {
        Node root = new Node();
        Tensor p = heap(2f);
        root.add("first", p);
        root.attach("child", new Node()).add("second", p);
        TrainingParameters source = root.trainingParameters();
        var original = p.hostStorage().orElseThrow();
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            assertThrows(IllegalArgumentException.class,
                    () -> TrainingParameterStorage.copyToSharedNative(source, arena));
            assertSame(original, p.hostStorage().orElseThrow());
            // Give the aliased Tensor valid native storage so identity uniqueness wins admission.
            p.replaceHostStorage(new MemorySegmentStorage(DataType.FLOAT32, 1,
                    arena.allocate(Float.BYTES, Float.BYTES)));
            assertThrows(IllegalArgumentException.class,
                    () -> TrainingSession.open(engine, source, p.mul(p), new Sgd(0.1)));
        }
    }

    @Test
    void realNnReplacementBeforeAndLateInRunOrRestoreNeverPublishesTrainingState() {
        for (GradientMode mode : GradientMode.values()) {
            try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
                Node root = new Node();
                Tensor p = heap(2f);
                Parameter wrapper = root.add("weight", p);
                TrainingParameters live = root.trainingParameters();
                TrainingParameterStorage.copyToSharedNative(live, arena);
                Tensor replacement = heap(3f);
                int[] validations = {0};
                int[] replaceAt = {0};
                TrainingParameters guarded = new TrainingParameters() {
                    @Override public List<Entry> entries() { return live.entries(); }
                    @Override public void validateUnchanged() {
                        if (++validations[0] == replaceAt[0]) wrapper.replace(replacement);
                        live.validateUnchanged();
                    }
                };
                try (TrainingSession session = TrainingSession.open(engine, guarded, p.mul(p),
                        new AdamW(0.01, 0.9, 0.999, 1e-8, 0))) {
                    TrainingState initial = session.state();
                    session.run(List.of(), GradientMode.RESET_AND_STEP);
                    session.run(List.of(), GradientMode.ACCUMULATE);
                    TrainingState before = session.state();
                    wrapper.replace(replacement);
                    assertThrows(IllegalStateException.class, session::state);
                    assertThrows(IllegalStateException.class, () -> session.restore(initial));
                    assertThrows(IllegalStateException.class, () -> session.run(List.of(), mode));
                    wrapper.replace(p);
                    assertState(before, session.state());
                    replaceAt[0] = validations[0] + 2; // after execution, at complete-view precommit
                    assertThrows(IllegalStateException.class, () -> session.run(List.of(), mode));
                    assertSame(replacement, wrapper.value());
                    assertEquals(3f, replacement.hostStorage().orElseThrow().segment()
                            .get(ValueLayout.JAVA_FLOAT_UNALIGNED, 0));
                    wrapper.replace(p);
                    assertState(before, session.state());
                    replaceAt[0] = validations[0] + 2; // restore's second complete-view check
                    assertThrows(IllegalStateException.class, () -> session.restore(initial));
                    wrapper.replace(p);
                    assertState(before, session.state());
                }
            }
        }
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> declarations(Node node) throws ReflectiveOperationException {
        Field field = Module.class.getDeclaredField("parameterDeclarations");
        field.setAccessible(true);
        return (Map<String, Object>) field.get(node);
    }

    private static void assertState(TrainingState expected, TrainingState actual) {
        assertEquals(expected.executions(), actual.executions());
        assertEquals(expected.optimizerSteps(), actual.optimizerSteps());
        assertEquals(expected.accumulatedGradientRuns(), actual.accumulatedGradientRuns());
        assertEquals(expected.optimizer(), actual.optimizer());
        assertEquals(expected.parameterGroups(), actual.parameterGroups());
        assertEquals(expected.parameters().size(), actual.parameters().size());
        for (int index = 0; index < expected.parameters().size(); index++) {
            var a = expected.parameters().get(index);
            var b = actual.parameters().get(index);
            assertEquals(a.path(), b.path());
            assertEquals(a.parameterBytes(), b.parameterBytes());
            assertEquals(a.masterParameterBytes(), b.masterParameterBytes());
            assertEquals(a.firstMomentBytes(), b.firstMomentBytes());
            assertEquals(a.secondMomentBytes(), b.secondMomentBytes());
            assertEquals(a.accumulatedGradientBytes(), b.accumulatedGradientBytes());
        }
    }

    private static Tensor heap(float value) { return TensorFactory.scalar(value, Optional.empty(), true); }

    private static final class Node extends Module {
        private Parameter add(String path, Tensor value) { return parameter(path, value); }
        private <T extends Module> T attach(String path, T node) { return child(path, node); }
    }
}
