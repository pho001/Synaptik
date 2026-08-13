package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.reflect.Method;
import java.util.List;
import org.junit.jupiter.api.Test;

class ModuleTest {
    @Test
    void retainsDirectDeclarationsInSeparateListsAndOneSharedNameSpace() {
        Tensor weight = tensor(1.0d);
        Tensor runningMean = tensor(0.0d);
        DeclaringModule module = new DeclaringModule(weight, runningMean);

        assertEquals(List.of(module.weight), module.parameters());
        assertEquals(List.of(module.runningMean), module.buffers());
        assertSame(weight, module.weight.value());
        assertSame(runningMean, module.runningMean.value());
        assertThrows(UnsupportedOperationException.class, () -> module.parameters().add(module.weight));
        assertThrows(UnsupportedOperationException.class, () -> module.buffers().add(module.runningMean));

        IllegalArgumentException parameterDuplicate = assertThrows(
                IllegalArgumentException.class, () -> module.declareParameter("weight", tensor(2.0d)));
        IllegalArgumentException bufferDuplicate = assertThrows(
                IllegalArgumentException.class, () -> module.declareBuffer("runningMean", tensor(2.0d)));
        IllegalArgumentException crossParameterDuplicate = assertThrows(
                IllegalArgumentException.class, () -> module.declareParameter("runningMean", tensor(2.0d)));
        IllegalArgumentException crossBufferDuplicate = assertThrows(
                IllegalArgumentException.class, () -> module.declareBuffer("weight", tensor(2.0d)));

        assertTrue(parameterDuplicate.getMessage().contains("already declared"));
        assertTrue(bufferDuplicate.getMessage().contains("already declared"));
        assertTrue(crossParameterDuplicate.getMessage().contains("already declared"));
        assertTrue(crossBufferDuplicate.getMessage().contains("already declared"));
    }

    @Test
    void rejectsNullAndBlankDirectNamesWithoutInstallingState() {
        DeclaringModule module = new DeclaringModule(tensor(1.0d), tensor(0.0d));

        assertThrows(NullPointerException.class, () -> module.declareParameter(null, tensor(2.0d)));
        assertThrows(NullPointerException.class, () -> module.declareBuffer(null, tensor(2.0d)));
        assertThrows(IllegalArgumentException.class, () -> module.declareParameter("   ", tensor(2.0d)));
        assertThrows(IllegalArgumentException.class, () -> module.declareBuffer("\t", tensor(2.0d)));
        assertThrows(IllegalArgumentException.class, () -> module.declareParameter("nested.weight", tensor(2.0d)));
        assertThrows(IllegalArgumentException.class, () -> module.declareBuffer("nested.runningMean", tensor(2.0d)));
        assertThrows(NullPointerException.class, () -> module.declareParameter("newParameter", null));
        assertThrows(NullPointerException.class, () -> module.declareBuffer("newBuffer", null));

        assertEquals(List.of(module.weight), module.parameters());
        assertEquals(List.of(module.runningMean), module.buffers());
    }

    @Test
    void declaresNoGenericForwardContract() {
        List<Method> forwardMethods = List.of(Module.class.getDeclaredMethods()).stream()
                .filter(method -> method.getName().equals("forward"))
                .toList();

        assertEquals(List.of(), forwardMethods);
    }

    private static Tensor tensor(double value) {
        return TensorFactory.scalar(value, java.util.Optional.empty(), true);
    }

    private static final class DeclaringModule extends Module {
        private final Parameter weight;
        private final Buffer runningMean;

        private DeclaringModule(Tensor weight, Tensor runningMean) {
            this.weight = parameter("weight", weight);
            this.runningMean = buffer("runningMean", runningMean);
        }

        private Parameter declareParameter(String name, Tensor value) {
            return parameter(name, value);
        }

        private Buffer declareBuffer(String name, Tensor value) {
            return buffer(name, value);
        }
    }
}
