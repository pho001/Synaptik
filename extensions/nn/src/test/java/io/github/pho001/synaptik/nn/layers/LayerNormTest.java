package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.normalization.AffineLayerNormAttrs;
import io.github.pho001.synaptik.model.operation.normalization.LayerNormKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class LayerNormTest {
    @Test
    void exposesExactlyThePlannedFinalPublicSurface() throws ReflectiveOperationException {
        Set<List<Class<?>>> constructors = Arrays.stream(LayerNorm.class.getDeclaredConstructors())
                .filter(constructor -> Modifier.isPublic(constructor.getModifiers()))
                .map(Constructor::getParameterTypes)
                .map(List::of)
                .collect(Collectors.toSet());
        Set<String> methods = Arrays.stream(LayerNorm.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(Method::getName)
                .collect(Collectors.toSet());
        Method scale = LayerNorm.class.getDeclaredMethod("scale");
        Method bias = LayerNorm.class.getDeclaredMethod("bias");
        Method forward = LayerNorm.class.getDeclaredMethod("forward", Tensor.class);

        assertAll(
                () -> assertTrue(Modifier.isPublic(LayerNorm.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(LayerNorm.class.getModifiers())),
                () -> assertSame(Module.class, LayerNorm.class.getSuperclass()),
                () -> assertEquals(
                        Set.of(
                                List.of(Tensor.class, Tensor.class, ScalarValue.class),
                                List.of(Shape.class, DataType.class, ScalarValue.class)),
                        constructors),
                () -> assertEquals(Set.of("scale", "bias", "forward"), methods),
                () -> assertSame(Parameter.class, scale.getReturnType()),
                () -> assertSame(Parameter.class, bias.getReturnType()),
                () -> assertSame(Tensor.class, forward.getReturnType()),
                () -> assertEquals(3, Arrays.stream(LayerNorm.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .count()),
                () -> assertTrue(Arrays.stream(LayerNorm.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .noneMatch(method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertFalse(Arrays.stream(Module.class.getDeclaredMethods())
                        .anyMatch(method -> method.getName().equals("forward"))));
    }

    @Test
    void suppliedConstructionRetainsExactStateNamesOrderReferencesAndWrappers() {
        Shape normalizedShape = Shape.of(3, 4);
        Tensor scale = tensor(DataType.FLOAT32, normalizedShape, true);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(3, 4), true);
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);

        LayerNorm layer = new LayerNorm(scale, bias, epsilon);

        assertAll(
                () -> assertSame(scale, layer.scale().value()),
                () -> assertSame(bias, layer.bias().value()),
                () -> assertEquals(List.of("scale", "bias"), names(layer.parameters())),
                () -> assertEquals(
                        List.of("scale", "bias"),
                        List.copyOf(layer.parametersRecursively().keySet())),
                () -> assertSame(layer.scale(), layer.parametersRecursively().get("scale")),
                () -> assertSame(layer.bias(), layer.parametersRecursively().get("bias")));

        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), false);
        AffineLayerNormAttrs attrs = attrs(layer.forward(input));
        assertAll(
                () -> assertSame(normalizedShape, attrs.normalizedShape()),
                () -> assertSame(epsilon, attrs.epsilon()));
    }

    @Test
    void suppliedConstructionValidatesInSpecifiedOrderBeforeDeclaration()
            throws ReflectiveOperationException {
        Tensor integralScale = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor noGradientScale = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor scalarScale = tensor(DataType.FLOAT32, Shape.scalar(), true);
        Tensor dynamicScale = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(2)),
                true);
        Tensor zeroFirstScale = tensor(DataType.FLOAT32, Shape.of(0, 0), true);
        Tensor zeroSecondScale = tensor(DataType.FLOAT32, Shape.of(2, 0), true);
        Tensor validScale = tensor(DataType.FLOAT32, Shape.of(2), true);
        Tensor integralBias = tensor(DataType.INT64, Shape.of(2), false);
        Tensor noGradientBias = tensor(DataType.FLOAT32, Shape.of(2), false);
        Tensor wrongTypeBias = tensor(DataType.FLOAT64, Shape.of(2), true);
        Tensor wrongShapeBias = tensor(DataType.FLOAT32, Shape.of(3), true);
        Tensor validBias = tensor(DataType.FLOAT32, Shape.of(2), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertAll(
                () -> assertEquals(
                        "scale",
                        assertThrows(
                                        NullPointerException.class,
                                        () -> new LayerNorm((Tensor) null, null, null))
                                .getMessage()),
                () -> assertEquals(
                        "bias",
                        assertThrows(
                                        NullPointerException.class,
                                        () -> new LayerNorm(integralScale, null, null))
                                .getMessage()),
                () -> assertEquals(
                        "epsilon",
                        assertThrows(
                                        NullPointerException.class,
                                        () -> new LayerNorm(integralScale, integralBias, null))
                                .getMessage()),
                () -> assertTrue(failure(integralScale, validBias, epsilon())
                        .contains("floating data type")),
                () -> assertTrue(failure(noGradientScale, validBias, epsilon())
                        .contains("requiresGrad")),
                () -> assertTrue(failure(scalarScale, validBias, epsilon())
                        .contains("positive rank")),
                () -> assertTrue(failure(dynamicScale, validBias, epsilon())
                        .contains("fully static")),
                () -> assertTrue(failure(zeroFirstScale, validBias, epsilon())
                        .contains("axis 0")),
                () -> assertTrue(failure(zeroSecondScale, validBias, epsilon())
                        .contains("axis 1")),
                () -> assertTrue(failure(validScale, integralBias, epsilon())
                        .contains("floating data type")),
                () -> assertTrue(failure(validScale, noGradientBias, epsilon())
                        .contains("requiresGrad")),
                () -> assertTrue(failure(validScale, wrongTypeBias, epsilon())
                        .contains("data type")),
                () -> assertTrue(failure(validScale, wrongShapeBias, epsilon())
                        .contains("Shape")),
                () -> assertTrue(failure(validScale, validBias, ScalarValue.float32(0.0f))
                        .contains("finite and positive")),
                () -> assertTrue(failure(validScale, validBias, ScalarValue.float64(1.0e-5))
                        .contains("epsilon data type")),
                () -> assertEquals(before, next.get()));
    }

    @Test
    void forwardDelegatesToExactAffineProducerAndIsModeInsensitive() {
        Shape normalizedShape = Shape.of(3, 4);
        Tensor scale = tensor(DataType.FLOAT32, normalizedShape, true);
        Tensor bias = tensor(DataType.FLOAT32, normalizedShape, true);
        Tensor input = tensor(DataType.BFLOAT16, Shape.of(2, 3, 4), false);
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        LayerNorm layer = new LayerNorm(scale, bias, epsilon);

        layer.eval();
        Tensor evaluation = layer.forward(input);
        layer.train();
        Tensor training = layer.forward(input);

        assertAffineExpression(evaluation, input, scale, bias, normalizedShape, epsilon);
        assertAffineExpression(training, input, scale, bias, normalizedShape, epsilon);
        assertAll(
                () -> assertEquals(ForwardMode.TRAINING, layer.mode()),
                () -> assertEquals(evaluation.descriptor(), training.descriptor()),
                () -> assertSame(DataType.FLOAT32, evaluation.descriptor().dataType()),
                () -> assertSame(input.descriptor().shape(), evaluation.descriptor().shape()),
                () -> assertTrue(evaluation.descriptor().requiresGrad()),
                () -> assertTrue(evaluation.descriptor().layout().isEmpty()),
                () -> assertTrue(evaluation.hostStorage().isEmpty()));
    }

    @Test
    void replacementChangesOnlyLaterForwardSnapshots() {
        Shape normalizedShape = Shape.of(3);
        Tensor oldScale = tensor(DataType.FLOAT32, normalizedShape, true);
        Tensor oldBias = tensor(DataType.FLOAT32, normalizedShape, true);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        LayerNorm layer = new LayerNorm(oldScale, oldBias, epsilon);
        Parameter scaleHandle = layer.scale();
        Parameter biasHandle = layer.bias();

        Tensor before = layer.forward(input);
        Tensor newScale = tensor(DataType.FLOAT32, Shape.of(3), true);
        Tensor newBias = tensor(DataType.FLOAT32, Shape.of(3), true);
        scaleHandle.replace(newScale);
        biasHandle.replace(newBias);
        Tensor after = layer.forward(input);

        assertAffineExpression(before, input, oldScale, oldBias, normalizedShape, epsilon);
        assertAffineExpression(after, input, newScale, newBias, normalizedShape, epsilon);
        assertAll(
                () -> assertSame(scaleHandle, layer.scale()),
                () -> assertSame(biasHandle, layer.bias()),
                () -> assertSame(newScale, layer.scale().value()),
                () -> assertSame(newBias, layer.bias().value()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> scaleHandle.replace(
                                tensor(DataType.FLOAT32, Shape.of(4), true))),
                () -> assertSame(newScale, scaleHandle.value()));
    }

    @Test
    void forwardRejectsNullAndInheritsModelFailuresAndIdentifierExhaustion()
            throws ReflectiveOperationException {
        Shape normalizedShape = Shape.of(3);
        LayerNorm layer = new LayerNorm(
                tensor(DataType.FLOAT32, normalizedShape, true),
                tensor(DataType.FLOAT32, normalizedShape, true),
                ScalarValue.float32(1.0e-5f));
        Tensor wrongTrailingShape = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor integralInput = tensor(DataType.INT32, Shape.of(2, 3), false);
        Tensor promotedInput = tensor(DataType.FLOAT64, Shape.of(2, 3), false);
        Tensor validInput = tensor(DataType.FLOAT32, Shape.of(2, 3), false);

        assertAll(
                () -> assertEquals(
                        "input",
                        assertThrows(NullPointerException.class, () -> layer.forward(null))
                                .getMessage()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> layer.forward(wrongTrailingShape)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> layer.forward(integralInput)),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> layer.forward(promotedInput))
                        .getMessage()
                        .contains("epsilon data type")));

        AtomicLong next = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long saved = next.get();
        boolean savedMaximumClaimed = maximumClaimed.get();
        try {
            next.set(Long.MAX_VALUE);
            maximumClaimed.set(true);
            IllegalStateException exhausted = assertThrows(
                    IllegalStateException.class,
                    () -> layer.forward(validInput));
            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", exhausted.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()));
        } finally {
            next.set(saved);
            maximumClaimed.set(savedMaximumClaimed);
        }
    }

    private static String failure(Tensor scale, Tensor bias, ScalarValue epsilon) {
        return assertThrows(
                        IllegalArgumentException.class,
                        () -> new LayerNorm(scale, bias, epsilon))
                .getMessage();
    }

    private static void assertAffineExpression(
            Tensor result,
            Tensor input,
            Tensor scale,
            Tensor bias,
            Shape normalizedShape,
            ScalarValue epsilon) {
        TensorProvenance provenance = result.provenance().orElseThrow();
        AffineLayerNormAttrs attrs = (AffineLayerNormAttrs) provenance.operation().attrs();
        assertAll(
                () -> assertSame(LayerNormKind.LAYER_NORM, provenance.operation().kind()),
                () -> assertSame(normalizedShape, attrs.normalizedShape()),
                () -> assertSame(epsilon, attrs.epsilon()),
                () -> assertEquals(List.of(input, scale, bias), provenance.inputs()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertEquals(1, provenance.producer().outputCount()),
                () -> assertSame(result.descriptor(), provenance.outputDescriptor()));
    }

    private static AffineLayerNormAttrs attrs(Tensor result) {
        return (AffineLayerNormAttrs) result.provenance().orElseThrow().operation().attrs();
    }

    private static List<String> names(List<Parameter> parameters) {
        return parameters.stream().map(Parameter::name).toList();
    }

    private static ScalarValue epsilon() {
        return ScalarValue.float32(1.0e-5f);
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

    private static AtomicBoolean maximumTensorIdClaimedState()
            throws ReflectiveOperationException {
        Field field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }
}
