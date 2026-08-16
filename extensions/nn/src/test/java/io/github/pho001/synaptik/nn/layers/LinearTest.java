package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import io.github.pho001.synaptik.nn.initialization.LinearWeightInitialization;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import io.github.pho001.synaptik.nn.module.Parameter;
import io.github.pho001.synaptik.nn.module.UnaryTensorModule;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class LinearTest {
    @Test
    void exposesExactlyThePlannedFinalPublicSurface() throws ReflectiveOperationException {
        Set<List<Class<?>>> constructors = Arrays.stream(Linear.class.getDeclaredConstructors())
                .filter(constructor -> Modifier.isPublic(constructor.getModifiers()))
                .map(Constructor::getParameterTypes)
                .map(List::of)
                .collect(Collectors.toSet());
        Set<String> methods = Arrays.stream(Linear.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(Method::getName)
                .collect(Collectors.toSet());
        Method weight = Linear.class.getDeclaredMethod("weight");
        Method bias = Linear.class.getDeclaredMethod("bias");
        Method forward = Linear.class.getDeclaredMethod("forward", Tensor.class);

        assertAll(
                () -> assertTrue(Modifier.isPublic(Linear.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(Linear.class.getModifiers())),
                () -> assertSame(UnaryTensorModule.class, Linear.class.getSuperclass()),
                () -> assertEquals(
                        Set.of(
                                List.of(Tensor.class),
                                List.of(Tensor.class, Tensor.class),
                                List.of(
                                        long.class,
                                        long.class,
                                        boolean.class,
                                        DataType.class,
                                        RandomGenerator.class),
                                List.of(
                                        long.class,
                                        boolean.class,
                                        DataType.class,
                                        LinearWeightInitialization.class,
                                        RandomGeneratorFactory.class,
                                        long.class)),
                        constructors),
                () -> assertEquals(Set.of("weight", "bias", "forward"), methods),
                () -> assertSame(Parameter.class, weight.getReturnType()),
                () -> assertSame(Optional.class, bias.getReturnType()),
                () -> assertSame(Tensor.class, forward.getReturnType()),
                () -> assertEquals(3, Arrays.stream(Linear.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .count()),
                () -> assertTrue(Arrays.stream(Linear.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .noneMatch(method -> Modifier.isStatic(method.getModifiers()))),
                () -> assertArrayEquals(
                        new LinearWeightInitialization[] {
                            LinearWeightInitialization.GLOROT_NORMAL,
                            LinearWeightInitialization.GLOROT_UNIFORM,
                            LinearWeightInitialization.KAIMING_RELU_NORMAL,
                            LinearWeightInitialization.KAIMING_RELU_UNIFORM
                        },
                        LinearWeightInitialization.values()),
                () -> assertFalse(Arrays.stream(UnaryTensorModule.class.getSuperclass().getDeclaredMethods())
                        .anyMatch(method -> method.getName().equals("forward"))));
    }

    @Test
    void suppliedConstructorsRetainExactStateNamesOrderAndWrappers() {
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(4, 3), true);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(4), true);

        Linear noBias = new Linear(weight);
        Linear biased = new Linear(weight, bias);

        assertAll(
                () -> assertSame(weight, noBias.weight().value()),
                () -> assertTrue(noBias.bias().isEmpty()),
                () -> assertEquals(List.of("weight"), names(noBias.parameters())),
                () -> assertEquals(
                        List.of("weight"), List.copyOf(noBias.parametersRecursively().keySet())),
                () -> assertSame(weight, biased.weight().value()),
                () -> assertSame(bias, biased.bias().orElseThrow().value()),
                () -> assertEquals(List.of("weight", "bias"), names(biased.parameters())),
                () -> assertEquals(
                        List.of("weight", "bias"),
                        List.copyOf(biased.parametersRecursively().keySet())),
                () -> assertSame(
                        biased.weight(), biased.parametersRecursively().get("weight")),
                () -> assertSame(
                        biased.bias().orElseThrow(),
                        biased.parametersRecursively().get("bias")));
    }

    @Test
    void suppliedConstructionValidatesInTheSpecifiedOrderBeforeDeclaration()
            throws ReflectiveOperationException {
        Tensor integralWeight = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor noGradientWeight = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor scalarWeight = tensor(DataType.FLOAT32, Shape.scalar(), true);
        Tensor dynamicWeight = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("N"), new StaticDimension(3)),
                true);
        Tensor zeroOutWeight = tensor(DataType.FLOAT32, Shape.of(0, 0), true);
        Tensor zeroInWeight = tensor(DataType.FLOAT32, Shape.of(4, 0), true);
        Tensor validWeight = tensor(DataType.FLOAT32, Shape.of(4, 3), true);
        Tensor integralBias = tensor(DataType.INT32, Shape.scalar(), false);
        Tensor noGradientBias = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor scalarBias = tensor(DataType.FLOAT32, Shape.scalar(), true);
        Tensor dynamicBias = tensor(
                DataType.FLOAT32, Shape.ofDimensions(new DynamicDimension("N")), true);
        Tensor wrongTypeBias = tensor(DataType.FLOAT64, Shape.of(4), true);
        Tensor wrongShapeBias = tensor(DataType.FLOAT32, Shape.of(5), true);
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        assertAll(
                () -> assertEquals(
                        "weight",
                        assertThrows(NullPointerException.class, () -> new Linear((Tensor) null))
                                .getMessage()),
                () -> assertEquals(
                        "bias",
                        assertThrows(
                                        NullPointerException.class,
                                        () -> new Linear(integralWeight, null))
                                .getMessage()),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(integralWeight))
                        .getMessage()
                        .contains("floating")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(noGradientWeight))
                        .getMessage()
                        .contains("requiresGrad")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(scalarWeight))
                        .getMessage()
                        .contains("rank two")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(dynamicWeight))
                        .getMessage()
                        .contains("fully static")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(zeroOutWeight))
                        .getMessage()
                        .contains("outFeatures")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(zeroInWeight))
                        .getMessage()
                        .contains("inFeatures")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(validWeight, integralBias))
                        .getMessage()
                        .contains("floating")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(validWeight, noGradientBias))
                        .getMessage()
                        .contains("requiresGrad")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(validWeight, scalarBias))
                        .getMessage()
                        .contains("rank one")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(validWeight, dynamicBias))
                        .getMessage()
                        .contains("fully static")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(validWeight, wrongTypeBias))
                        .getMessage()
                        .contains("data type")),
                () -> assertTrue(assertThrows(
                                IllegalArgumentException.class,
                                () -> new Linear(validWeight, wrongShapeBias))
                        .getMessage()
                        .contains("outFeatures")),
                () -> assertEquals(before, next.get()),
                () -> assertTrue(validWeight.provenance().isEmpty()),
                () -> assertTrue(wrongShapeBias.provenance().isEmpty()));
    }

    @Test
    void forwardDelegatesToExactPrimitiveChainsAndIsModeInsensitive() {
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(4, 3), true);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(4), true);
        Tensor input = tensor(DataType.FLOAT64, Shape.of(2, 3), true);
        Linear noBias = new Linear(weight);
        Linear biased = new Linear(weight, bias);

        noBias.eval();
        Tensor evaluation = noBias.forward(input);
        noBias.train();
        Tensor training = noBias.forward(input);
        Tensor biasedResult = biased.forward(input);

        assertNoBiasChain(evaluation, input, weight);
        assertNoBiasChain(training, input, weight);
        assertBiasedChain(biasedResult, input, weight, bias);
        assertAll(
                () -> assertEquals(ForwardMode.TRAINING, noBias.mode()),
                () -> assertEquals(evaluation.descriptor(), training.descriptor()),
                () -> assertSame(DataType.FLOAT64, biasedResult.descriptor().dataType()),
                () -> assertEquals(Shape.of(2, 4), biasedResult.descriptor().shape()),
                () -> assertTrue(biasedResult.descriptor().requiresGrad()),
                () -> assertTrue(biasedResult.hostStorage().isEmpty()));
    }

    @Test
    void replacementChangesOnlyLaterForwardSnapshots() {
        Tensor oldWeight = tensor(DataType.FLOAT32, Shape.of(4, 3), true);
        Tensor oldBias = tensor(DataType.FLOAT32, Shape.of(4), true);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Linear layer = new Linear(oldWeight, oldBias);
        Parameter weightHandle = layer.weight();
        Parameter biasHandle = layer.bias().orElseThrow();

        Tensor before = layer.forward(input);
        Tensor newWeight = tensor(DataType.FLOAT32, Shape.of(4, 3), true);
        Tensor newBias = tensor(DataType.FLOAT32, Shape.of(4), true);
        weightHandle.replace(newWeight);
        biasHandle.replace(newBias);
        Tensor after = layer.forward(input);

        assertBiasedChain(before, input, oldWeight, oldBias);
        assertBiasedChain(after, input, newWeight, newBias);
        assertAll(
                () -> assertSame(weightHandle, layer.weight()),
                () -> assertSame(biasHandle, layer.bias().orElseThrow()),
                () -> assertSame(newWeight, layer.weight().value()),
                () -> assertSame(newBias, layer.bias().orElseThrow().value()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> weightHandle.replace(
                                tensor(DataType.FLOAT32, Shape.of(3, 4), true))),
                () -> assertSame(newWeight, weightHandle.value()));
    }

    @Test
    void forwardRejectsNullFirstAndInheritsModelRankAndContractionFailures() {
        Linear layer = new Linear(tensor(DataType.FLOAT32, Shape.of(4, 3), true));
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor wrongFeatures = tensor(DataType.FLOAT32, Shape.of(2, 5), false);

        NullPointerException nullInput = assertThrows(
                NullPointerException.class, () -> layer.forward(null));

        assertAll(
                () -> assertEquals("input", nullInput.getMessage()),
                () -> assertThrows(IllegalArgumentException.class, () -> layer.forward(scalar)),
                () -> assertThrows(
                        IllegalArgumentException.class, () -> layer.forward(wrongFeatures)));
    }

    @Test
    void automaticConstructorInfersOnlyFinalFeaturesAndKeepsLeadingDimensionsVariable() {
        Linear layer = new Linear(
                4,
                true,
                DataType.FLOAT32,
                LinearWeightInitialization.GLOROT_UNIFORM,
                RandomGeneratorFactory.of("L64X128MixRandom"),
                41L);
        Linear noBias = new Linear(
                4,
                false,
                DataType.FLOAT32,
                LinearWeightInitialization.GLOROT_UNIFORM,
                RandomGeneratorFactory.of("L64X128MixRandom"),
                42L);
        Tensor firstInput = tensor(DataType.FLOAT32, Shape.of(7, 3), false);

        assertThrows(IllegalStateException.class, layer::weight);
        assertThrows(IllegalStateException.class, layer::bias);
        assertThrows(IllegalStateException.class, layer::parameters);
        assertTrue(layer.buffers().isEmpty());
        assertTrue(noBias.bias().isEmpty());

        Tensor first = layer.forward(firstInput);
        Parameter weight = layer.weight();
        Parameter bias = layer.bias().orElseThrow();
        Tensor rankThreeInput = tensor(DataType.FLOAT32, Shape.of(2, 5, 3), false);
        Tensor dynamicLeadingInput = tensor(
                DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("B"), new StaticDimension(3)),
                false);
        Tensor rankThree = layer.forward(rankThreeInput);
        Tensor dynamicLeading = layer.forward(dynamicLeadingInput);

        assertAll(
                () -> assertEquals(Shape.of(4, 3), weight.value().descriptor().shape()),
                () -> assertEquals(Shape.of(4), bias.value().descriptor().shape()),
                () -> assertBiasedChain(first, firstInput, weight.value(), bias.value()),
                () -> assertEquals(Shape.of(2, 5, 4), rankThree.descriptor().shape()),
                () -> assertEquals(
                        Shape.ofDimensions(new DynamicDimension("B"), new StaticDimension(4)),
                        dynamicLeading.descriptor().shape()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> layer.forward(tensor(DataType.FLOAT32, Shape.of(2, 5), false))),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> layer.forward(tensor(DataType.FLOAT64, Shape.of(2, 3), false))),
                () -> assertSame(weight, layer.weight()),
                () -> assertSame(bias, layer.bias().orElseThrow()));
    }

    private static void assertNoBiasChain(Tensor result, Tensor input, Tensor weight) {
        TensorProvenance matmul = result.provenance().orElseThrow();
        Tensor transposedWeight = matmul.inputs().get(1);
        TensorProvenance permute = transposedWeight.provenance().orElseThrow();
        assertAll(
                () -> assertSame(MatmulKind.MATMUL, matmul.operation().kind()),
                () -> assertEquals(0, matmul.outputIndex()),
                () -> assertSame(input, matmul.inputs().get(0)),
                () -> assertSame(AxisTransformKind.PERMUTE, permute.operation().kind()),
                () -> assertEquals(
                        List.of(1, 0),
                        ((PermutationAttrs) permute.operation().attrs()).axes()),
                () -> assertEquals(0, permute.outputIndex()),
                () -> assertSame(weight, permute.inputs().getFirst()));
    }

    private static void assertBiasedChain(
            Tensor result, Tensor input, Tensor weight, Tensor bias) {
        TensorProvenance add = result.provenance().orElseThrow();
        Tensor product = add.inputs().getFirst();
        assertAll(
                () -> assertSame(BinaryArithmeticKind.ADD, add.operation().kind()),
                () -> assertEquals(0, add.outputIndex()),
                () -> assertSame(bias, add.inputs().get(1)));
        assertNoBiasChain(product, input, weight);
    }

    private static List<String> names(List<Parameter> parameters) {
        return parameters.stream().map(Parameter::name).toList();
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
}
