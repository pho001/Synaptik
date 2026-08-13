package io.github.pho001.synaptik.nn.consumer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class ParameterUpdateContractTest {
    @Test
    void genericConsumerReplacesEveryDiscoveredParameterWithoutConcreteModuleKnowledge() {
        TestModule child = new TestModule(
                "weight", floating(Shape.of(2), DataType.FLOAT64, true));
        TestModule rootModule = new TestModule(
                "rootWeight", floating(Shape.of(3), DataType.FLOAT32, true));
        rootModule.attach("child", child);
        Module root = rootModule;

        Map<String, Tensor> replacements = replaceEveryParameter(root);
        Map<String, Parameter> discoveredAfter = root.parametersRecursively();

        assertEquals(
                List.of("rootWeight", "child.weight"),
                List.copyOf(discoveredAfter.keySet()));
        assertSame(rootModule.parameter, discoveredAfter.get("rootWeight"));
        assertSame(child.parameter, discoveredAfter.get("child.weight"));
        assertSame(replacements.get("rootWeight"), rootModule.parameter.value());
        assertSame(replacements.get("child.weight"), child.parameter.value());
    }

    @Test
    void downstreamReplacementRejectsNullTypeShapeAndGradientMismatchesWithoutMutation() {
        TestModule concreteRoot = new TestModule(
                "weight", floating(Shape.of(2), DataType.FLOAT64, true));
        Module root = concreteRoot;
        Parameter parameter = root.parametersRecursively().get("weight");
        Tensor original = parameter.value();

        assertThrows(NullPointerException.class, () -> parameter.replace(null));
        assertSame(original, parameter.value());

        assertThrows(
                IllegalArgumentException.class,
                () -> parameter.replace(floating(Shape.of(2), DataType.FLOAT32, true)));
        assertSame(original, parameter.value());

        assertThrows(
                IllegalArgumentException.class,
                () -> parameter.replace(floating(Shape.of(3), DataType.FLOAT64, true)));
        assertSame(original, parameter.value());

        assertThrows(
                IllegalArgumentException.class,
                () -> parameter.replace(floating(Shape.of(2), DataType.FLOAT64, false)));
        assertSame(original, parameter.value());

        Tensor compatible = floating(Shape.of(2), DataType.FLOAT64, true);
        parameter.replace(compatible);
        assertSame(compatible, parameter.value());
    }

    private static Map<String, Tensor> replaceEveryParameter(Module root) {
        Map<String, Tensor> replacements = new LinkedHashMap<>();
        root.parametersRecursively().forEach((path, parameter) -> {
            Tensor current = parameter.value();
            Tensor replacement = current.add(current);
            parameter.replace(replacement);
            replacements.put(path, replacement);
        });
        return replacements;
    }

    private static Tensor floating(Shape shape, DataType dataType, boolean requiresGrad) {
        return TensorFactory.zeros(shape, dataType, Optional.empty(), requiresGrad);
    }

    private static final class TestModule extends Module {
        private final Parameter parameter;

        private TestModule(String name, Tensor value) {
            parameter = parameter(name, value);
        }

        private void attach(String name, Module child) {
            child(name, child);
        }
    }
}
