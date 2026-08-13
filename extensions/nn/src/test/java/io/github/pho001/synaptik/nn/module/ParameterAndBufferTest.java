package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class ParameterAndBufferTest {
    @Test
    void parameterAndBufferClassifyExactTensorReferencesWithoutTensorInheritance() {
        Tensor parameterTensor = TensorFactory.scalar(1.0d, Optional.empty(), true);
        Tensor bufferTensor = TensorFactory.scalar(0.0d, Optional.empty(), false);
        StateModule module = new StateModule(parameterTensor, bufferTensor);

        assertEquals("weight", module.parameter.name());
        assertEquals("runningMean", module.buffer.name());
        assertSame(parameterTensor, module.parameter.value());
        assertSame(bufferTensor, module.buffer.value());
        assertFalse(Tensor.class.isAssignableFrom(Parameter.class));
        assertFalse(Tensor.class.isAssignableFrom(Buffer.class));
        assertTrue(module.parameter instanceof Parameter);
        assertTrue(module.buffer instanceof Buffer);
    }

    @Test
    void exposesNoBindingReplacementOrUpdateApi() {
        assertFalse(Arrays.stream(Parameter.class.getDeclaredMethods()).anyMatch(this::isPublicOrProtectedBindingMutation));
        assertFalse(Arrays.stream(Buffer.class.getDeclaredMethods()).anyMatch(this::isPublicOrProtectedBindingMutation));
        assertTrue(Arrays.stream(Parameter.class.getDeclaredFields())
                .filter(field -> field.getName().equals("name"))
                .allMatch(field -> Modifier.isFinal(field.getModifiers())));
        assertTrue(Arrays.stream(Buffer.class.getDeclaredFields())
                .filter(field -> field.getName().equals("name"))
                .allMatch(field -> Modifier.isFinal(field.getModifiers())));
    }

    private boolean isPublicOrProtectedBindingMutation(Method method) {
        return (Modifier.isPublic(method.getModifiers()) || Modifier.isProtected(method.getModifiers()))
                && (method.getName().contains("Value")
                || method.getName().equals("replace")
                || method.getName().equals("update")
                || method.getName().equals("rebind"));
    }

    private static final class StateModule extends Module {
        private final Parameter parameter;
        private final Buffer buffer;

        private StateModule(Tensor parameterTensor, Tensor bufferTensor) {
            parameter = parameter("weight", parameterTensor);
            buffer = buffer("runningMean", bufferTensor);
        }
    }
}
