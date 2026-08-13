package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class ModuleTreeTest {
    @Test
    void exposesDirectAndRecursiveSnapshotsInSpecifiedOrderWithDotPaths() {
        StateModule layer1 = new StateModule("weight", "runningMean");
        StateModule layer2 = new StateModule("weight", "runningMean");
        TreeModule encoder = new TreeModule();
        encoder.attach("layer1", layer1);
        encoder.attach("layer2", layer2);
        TreeModule model = new TreeModule("rootWeight", "rootBuffer");
        model.attach("encoder", encoder);

        assertEquals(List.of("encoder"), List.copyOf(model.children().keySet()));
        assertSame(encoder, model.children().get("encoder"));
        assertEquals(
                List.of("rootWeight", "encoder.layer1.weight", "encoder.layer2.weight"),
                List.copyOf(model.parametersRecursively().keySet()));
        assertEquals(
                List.of("rootBuffer", "encoder.layer1.runningMean", "encoder.layer2.runningMean"),
                List.copyOf(model.buffersRecursively().keySet()));
        assertSame(layer1.parameter, model.parametersRecursively().get("encoder.layer1.weight"));
        assertSame(layer2.buffer, model.buffersRecursively().get("encoder.layer2.runningMean"));
        assertThrows(UnsupportedOperationException.class, () -> model.children().clear());
        assertThrows(UnsupportedOperationException.class, () -> model.parametersRecursively().clear());
        assertThrows(UnsupportedOperationException.class, () -> model.buffersRecursively().clear());

        Map<String, Module> beforeRegistration = model.children();
        model.attach("later", new TreeModule());
        assertEquals(List.of("encoder"), List.copyOf(beforeRegistration.keySet()));
        assertEquals(List.of("encoder", "later"), List.copyOf(model.children().keySet()));
    }

    @Test
    void sharesOneNamespaceAcrossStateAndChildren() {
        TreeModule module = new TreeModule("weight", "runningMean");

        assertThrows(IllegalArgumentException.class, () -> module.attach("weight", new TreeModule()));
        assertThrows(IllegalArgumentException.class, () -> module.attach("runningMean", new TreeModule()));
        module.attach("encoder", new TreeModule());
        assertThrows(IllegalArgumentException.class, () -> module.declareParameter("encoder"));
        assertThrows(IllegalArgumentException.class, () -> module.declareBuffer("encoder"));
        assertThrows(IllegalArgumentException.class, () -> module.attach("encoder", new TreeModule()));

        assertEquals(List.of("encoder"), List.copyOf(module.children().keySet()));
        assertEquals(List.of("weight"), List.copyOf(module.parametersRecursively().keySet()));
        assertEquals(List.of("runningMean"), List.copyOf(module.buffersRecursively().keySet()));
    }

    @Test
    void rejectsInvalidOrAlreadyOwnedChildrenWithoutChangingOwnershipOrRegistries() {
        TreeModule parent = new TreeModule();
        TreeModule child = new TreeModule();
        TreeModule otherParent = new TreeModule();

        assertThrows(NullPointerException.class, () -> parent.attach(null, child));
        assertThrows(IllegalArgumentException.class, () -> parent.attach("  ", child));
        assertThrows(IllegalArgumentException.class, () -> parent.attach("nested.child", child));
        assertThrows(NullPointerException.class, () -> parent.attach("child", null));
        assertThrows(IllegalArgumentException.class, () -> parent.attach("self", parent));
        assertTrue(parent.children().isEmpty());

        parent.attach("child", child);
        assertThrows(IllegalStateException.class, () -> otherParent.attach("child", child));
        assertEquals(List.of("child"), List.copyOf(parent.children().keySet()));
        assertTrue(otherParent.children().isEmpty());

        assertThrows(IllegalArgumentException.class, () -> child.attach("ancestor", parent));
        assertTrue(child.children().isEmpty());
        assertEquals(List.of("child"), List.copyOf(parent.children().keySet()));
    }

    @Test
    void recursivelyPropagatesModesAndLeavesExistingContextsUnchanged() {
        TreeModule root = new TreeModule();
        TreeModule child = new TreeModule();
        TreeModule grandchild = new TreeModule();
        child.attach("grandchild", grandchild);
        root.attach("child", child);
        ForwardContext initial = grandchild.forwardContext();

        root.eval();
        assertEquals(ForwardMode.EVALUATION, root.mode());
        assertEquals(ForwardMode.EVALUATION, child.mode());
        assertEquals(ForwardMode.EVALUATION, grandchild.mode());
        assertEquals(ForwardMode.TRAINING, initial.mode());

        root.train();
        assertEquals(ForwardMode.TRAINING, root.mode());
        assertEquals(ForwardMode.TRAINING, child.mode());
        assertEquals(ForwardMode.TRAINING, grandchild.mode());
    }

    @Test
    void malformedRepeatedIdentityFailsBeforeChangingAnyMode() throws ReflectiveOperationException {
        TreeModule root = new TreeModule();
        TreeModule child = new TreeModule();
        root.attach("child", child);
        root.eval();
        corruptChildren(child).put("cycle", root);

        assertThrows(IllegalStateException.class, root::parametersRecursively);
        assertThrows(IllegalStateException.class, root::buffersRecursively);
        assertThrows(IllegalStateException.class, root::train);
        assertThrows(IllegalStateException.class, root::eval);
        assertEquals(ForwardMode.EVALUATION, root.mode());
        assertEquals(ForwardMode.EVALUATION, child.mode());
    }

    @Test
    void deepValidChainSupportsDiscoveryAndModeChangesWithoutCallStackDepthLimit() {
        int depth = 20_000;
        StateModule leaf = new StateModule("weight", "runningMean");
        Module root = leaf;
        List<Module> modules = new ArrayList<>(depth + 1);
        modules.add(leaf);
        for (int index = 0; index < depth; index++) {
            TreeModule parent = new TreeModule();
            parent.attach("next", root);
            root = parent;
            modules.add(parent);
        }
        String prefix = "next.".repeat(depth);

        Map<String, Parameter> parameters = root.parametersRecursively();
        Map<String, Buffer> buffers = root.buffersRecursively();

        assertEquals(List.of(prefix + "weight"), List.copyOf(parameters.keySet()));
        assertEquals(List.of(prefix + "runningMean"), List.copyOf(buffers.keySet()));
        assertSame(leaf.parameter, parameters.get(prefix + "weight"));
        assertSame(leaf.buffer, buffers.get(prefix + "runningMean"));

        root.eval();
        assertTrue(modules.stream().allMatch(
                module -> module.mode() == ForwardMode.EVALUATION));
        root.train();
        assertTrue(modules.stream().allMatch(
                module -> module.mode() == ForwardMode.TRAINING));
    }

    @Test
    void structuralSnapshotsRetainWrappersWhoseBindingsObserveLaterDirectReplacement() {
        StateModule child = new StateModule("weight", "runningMean");
        TreeModule root = new TreeModule("rootWeight", "rootBuffer");
        root.attach("child", child);
        Map<String, Parameter> parametersBefore = root.parametersRecursively();
        Map<String, Buffer> buffersBefore = root.buffersRecursively();
        Parameter childParameter = parametersBefore.get("child.weight");
        Buffer childBuffer = buffersBefore.get("child.runningMean");
        Tensor nextParameter = TensorFactory.scalar(2.0d, Optional.empty(), true);
        Tensor nextBuffer = TensorFactory.scalar(3.0d, Optional.empty(), true);

        child.replaceDirectParameter("weight", nextParameter);
        child.replaceDirectBuffer("runningMean", nextBuffer);

        assertSame(child.parameter, childParameter);
        assertSame(child.buffer, childBuffer);
        assertSame(nextParameter, childParameter.value());
        assertSame(nextBuffer, childBuffer.value());
        assertEquals(List.of("rootWeight", "child.weight"), List.copyOf(parametersBefore.keySet()));
        assertEquals(List.of("rootBuffer", "child.runningMean"), List.copyOf(buffersBefore.keySet()));
        assertThrows(UnsupportedOperationException.class, parametersBefore::clear);
        assertThrows(UnsupportedOperationException.class, buffersBefore::clear);
    }

    @Test
    void parentCannotReplaceAChildBindingByDotPath() {
        StateModule child = new StateModule("weight", "runningMean");
        TreeModule root = new TreeModule();
        root.attach("child", child);
        Tensor original = child.parameter.value();

        assertThrows(IllegalArgumentException.class, () -> root.replaceDirectParameter("child.weight", tensor()));
        assertThrows(IllegalArgumentException.class, () -> root.replaceDirectBuffer("child.runningMean", tensor()));

        assertSame(original, child.parameter.value());
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Module> corruptChildren(Module module) throws ReflectiveOperationException {
        Field children = Module.class.getDeclaredField("children");
        children.setAccessible(true);
        return (Map<String, Module>) children.get(module);
    }

    private static Tensor tensor() {
        return TensorFactory.scalar(1.0d, Optional.empty(), true);
    }

    private static final class TreeModule extends Module {
        private TreeModule() {
        }

        private TreeModule(String parameterName, String bufferName) {
            declareParameter(parameterName);
            declareBuffer(bufferName);
        }

        private <T extends Module> T attach(String name, T child) {
            return child(name, child);
        }

        private Parameter declareParameter(String name) {
            return parameter(name, tensor());
        }

        private Buffer declareBuffer(String name) {
            return buffer(name, tensor());
        }

        private void replaceDirectParameter(String name, Tensor value) {
            replaceParameter(name, value);
        }

        private void replaceDirectBuffer(String name, Tensor value) {
            replaceBuffer(name, value);
        }
    }

    private static final class StateModule extends Module {
        private final Parameter parameter;
        private final Buffer buffer;

        private StateModule(String parameterName, String bufferName) {
            parameter = parameter(parameterName, tensor());
            buffer = buffer(bufferName, tensor());
        }

        private void replaceDirectParameter(String name, Tensor value) {
            replaceParameter(name, value);
        }

        private void replaceDirectBuffer(String name, Tensor value) {
            replaceBuffer(name, value);
        }
    }
}
