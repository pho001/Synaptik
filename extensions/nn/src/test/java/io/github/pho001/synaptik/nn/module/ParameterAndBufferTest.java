package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Optional;
import java.util.Set;
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
    void exposesOnlyPublicParameterReplacementAndNoBufferReplacementApi()
            throws ReflectiveOperationException {
        Method replace = Parameter.class.getDeclaredMethod("replace", Tensor.class);

        assertTrue(Modifier.isPublic(replace.getModifiers()));
        assertEquals(void.class, replace.getReturnType());
        assertTrue(Modifier.isFinal(Parameter.class.getModifiers()));
        assertEquals(
                Set.of("name", "value", "replace"),
                Arrays.stream(Parameter.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .map(Method::getName)
                        .collect(java.util.stream.Collectors.toUnmodifiableSet()));
        assertEquals(1, Parameter.class.getDeclaredConstructors().length);
        assertFalse(Modifier.isPublic(
                Parameter.class.getDeclaredConstructors()[0].getModifiers()));
        assertFalse(Modifier.isProtected(
                Parameter.class.getDeclaredConstructors()[0].getModifiers()));
        assertEquals(
                1,
                Arrays.stream(Parameter.class.getDeclaredMethods())
                        .filter(this::isPublicOrProtectedBindingMutation)
                        .count());
        assertFalse(Arrays.stream(Buffer.class.getDeclaredMethods()).anyMatch(this::isPublicOrProtectedBindingMutation));
        assertTrue(Arrays.stream(Parameter.class.getDeclaredFields())
                .filter(field -> field.getName().equals("name"))
                .allMatch(field -> Modifier.isFinal(field.getModifiers())));
        assertTrue(Arrays.stream(Buffer.class.getDeclaredFields())
                .filter(field -> field.getName().equals("name"))
                .allMatch(field -> Modifier.isFinal(field.getModifiers())));
    }

    @Test
    void parameterDeclarationRequiresFloatingGradientEligibleValueInValidationOrder() {
        NullPointerException nameNull = assertThrows(
                NullPointerException.class, () -> new Parameter(null, null));
        NullPointerException valueNull = assertThrows(
                NullPointerException.class, () -> new Parameter("weight", null));
        IllegalArgumentException type = assertThrows(
                IllegalArgumentException.class,
                () -> new Parameter(
                        "weight",
                        TensorFactory.zeros(
                                Shape.of(2), DataType.INT32, Optional.empty(), false)));
        IllegalArgumentException gradient = assertThrows(
                IllegalArgumentException.class,
                () -> new Parameter(
                        "weight",
                        TensorFactory.zeros(
                                Shape.of(2), DataType.FLOAT64, Optional.empty(), false)));

        assertEquals("name", nameNull.getMessage());
        assertEquals("value", valueNull.getMessage());
        assertEquals(
                "parameter value must have a floating data type: INT32", type.getMessage());
        assertEquals(
                "parameter value must have requiresGrad == true", gradient.getMessage());
    }

    @Test
    void failedModuleDeclarationPreservesRegistryAndNamePreflightOrder() {
        StateModule module = new StateModule(
                TensorFactory.scalar(1.0d, Optional.empty(), true),
                TensorFactory.scalar(0.0d, Optional.empty(), false));
        Tensor integral = TensorFactory.zeros(
                Shape.of(2), DataType.INT32, Optional.empty(), false);
        Tensor noGradient = TensorFactory.zeros(
                Shape.of(2), DataType.FLOAT64, Optional.empty(), false);

        assertThrows(
                IllegalArgumentException.class,
                () -> module.declareParameter("integral", integral));
        assertThrows(
                IllegalArgumentException.class,
                () -> module.declareParameter("noGradient", noGradient));
        assertThrows(
                IllegalArgumentException.class,
                () -> module.declareParameter("weight", null));

        assertEquals(java.util.List.of(module.parameter), module.parameters());
        assertEquals(java.util.Set.of("weight"), module.parametersRecursively().keySet());
    }

    @Test
    void parameterReplacementPreservesSchemaAndAcceptsOtherTensorFacts() {
        Shape declaredShape = Shape.of(2);
        Tensor original = TensorFactory.zeros(
                declaredShape, DataType.FLOAT64, Optional.of("declared"), true);
        Parameter parameter = new Parameter("weight", original);
        Tensor expression = original.add(TensorFactory.ones(
                Shape.of(2), DataType.FLOAT64, Optional.empty(), true));

        parameter.replace(expression);

        assertSame(expression, parameter.value());
        assertTrue(expression.provenance().isPresent());
        assertTrue(expression.hostStorage().isEmpty());
        assertTrue(expression.descriptor().layout().isEmpty());
        assertTrue(expression.label().isEmpty());
        assertSame(original, expression.provenance().orElseThrow().inputs().getFirst());
    }

    @Test
    void parameterReplacementValidatesNullTypeShapeThenGradientWithoutMutation() {
        Shape declaredShape = Shape.of(2);
        Tensor original = TensorFactory.zeros(
                declaredShape, DataType.FLOAT64, Optional.empty(), true);
        Parameter parameter = new Parameter("weight", original);
        Tensor wrongType = TensorFactory.zeros(
                Shape.of(3), DataType.FLOAT32, Optional.empty(), false);
        Tensor wrongShape = TensorFactory.zeros(
                Shape.of(3), DataType.FLOAT64, Optional.empty(), false);
        Tensor noGradient = TensorFactory.zeros(
                Shape.of(2), DataType.FLOAT64, Optional.empty(), false);

        assertThrows(NullPointerException.class, () -> parameter.replace(null));
        assertSame(original, parameter.value());
        IllegalArgumentException type = assertThrows(
                IllegalArgumentException.class, () -> parameter.replace(wrongType));
        assertSame(original, parameter.value());
        IllegalArgumentException shape = assertThrows(
                IllegalArgumentException.class, () -> parameter.replace(wrongShape));
        assertSame(original, parameter.value());
        IllegalArgumentException gradient = assertThrows(
                IllegalArgumentException.class, () -> parameter.replace(noGradient));
        assertSame(original, parameter.value());
        assertTrue(type.getMessage().startsWith("replacement data type"));
        assertTrue(shape.getMessage().startsWith("replacement shape"));
        assertEquals(
                "replacement value must have requiresGrad == true", gradient.getMessage());

        Tensor structurallyCompatible = TensorFactory.zeros(
                Shape.of(2), DataType.FLOAT64, Optional.empty(), true);
        parameter.replace(structurallyCompatible);
        assertSame(structurallyCompatible, parameter.value());
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

        private Parameter declareParameter(String name, Tensor value) {
            return parameter(name, value);
        }
    }
}
