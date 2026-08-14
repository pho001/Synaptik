package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormInferenceAttrs;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormKind;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormTrainingAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import io.github.pho001.synaptik.nn.module.Buffer;
import io.github.pho001.synaptik.nn.module.ForwardContext;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class BatchNormTest {
    @Test
    void exposesExactlyThePlannedFinalPublicSurface() throws ReflectiveOperationException {
        Set<List<Class<?>>> constructors = Arrays.stream(BatchNorm.class.getDeclaredConstructors())
                .filter(constructor -> Modifier.isPublic(constructor.getModifiers()))
                .map(Constructor::getParameterTypes)
                .map(List::of)
                .collect(Collectors.toSet());
        List<Method> visibleMethods = Arrays.stream(BatchNorm.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers())
                        || Modifier.isProtected(method.getModifiers()))
                .toList();
        Set<String> methodNames = visibleMethods.stream()
                .map(Method::getName)
                .collect(Collectors.toSet());
        Set<String> visibleFields = Arrays.stream(BatchNorm.class.getDeclaredFields())
                .filter(field -> Modifier.isPublic(field.getModifiers())
                        || Modifier.isProtected(field.getModifiers()))
                .map(Field::getName)
                .collect(Collectors.toSet());
        Set<String> bufferVisibleMethods = Arrays.stream(Buffer.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers())
                        || Modifier.isProtected(method.getModifiers()))
                .map(Method::getName)
                .collect(Collectors.toSet());

        assertAll(
                () -> assertTrue(Modifier.isPublic(BatchNorm.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(BatchNorm.class.getModifiers())),
                () -> assertSame(Module.class, BatchNorm.class.getSuperclass()),
                () -> assertEquals(Set.of(
                                List.of(
                                        int.class,
                                        Tensor.class,
                                        Tensor.class,
                                        Tensor.class,
                                        Tensor.class,
                                        ScalarValue.class,
                                        ScalarValue.class),
                                List.of(
                                        long.class,
                                        int.class,
                                        DataType.class,
                                        ScalarValue.class,
                                        ScalarValue.class)),
                        constructors),
                () -> assertEquals(
                        Set.of("scale", "bias", "runningMean", "runningVariance", "forward"),
                        methodNames),
                () -> assertEquals(5, visibleMethods.size()),
                () -> assertTrue(visibleMethods.stream()
                        .allMatch(method -> Modifier.isPublic(method.getModifiers()))),
                () -> assertEquals(Parameter.class,
                        BatchNorm.class.getDeclaredMethod("scale").getReturnType()),
                () -> assertEquals(Parameter.class,
                        BatchNorm.class.getDeclaredMethod("bias").getReturnType()),
                () -> assertEquals(Buffer.class,
                        BatchNorm.class.getDeclaredMethod("runningMean").getReturnType()),
                () -> assertEquals(Buffer.class,
                        BatchNorm.class.getDeclaredMethod("runningVariance").getReturnType()),
                () -> assertEquals(Tensor.class,
                        BatchNorm.class.getDeclaredMethod(
                                "forward", Tensor.class, ForwardContext.class).getReturnType()),
                () -> assertTrue(visibleFields.isEmpty()),
                () -> assertEquals(Set.of("name", "value"), bufferVisibleMethods));
    }

    @Test
    void suppliedStateRetainsExactReferencesNamesSchemasAndStableDiscoveryWrappers()
            throws ReflectiveOperationException {
        Shape scaleShape = Shape.of(3);
        Tensor scale = tensor(DataType.FLOAT32, scaleShape, true);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(3), true);
        Tensor mean = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor variance = tensor(DataType.FLOAT32, Shape.of(3), false);
        ScalarValue momentum = ScalarValue.float32(0.25f);
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        AtomicLong next = nextTensorIdState();
        long beforeConstruction = next.get();

        BatchNorm layer = new BatchNorm(
                1, scale, bias, mean, variance, momentum, epsilon);
        Map<String, Parameter> parameterSnapshot = layer.parametersRecursively();
        Map<String, Buffer> bufferSnapshot = layer.buffersRecursively();

        assertAll(
                () -> assertEquals(List.of("scale", "bias"), names(layer.parameters())),
                () -> assertEquals(
                        List.of("runningMean", "runningVariance"), bufferNames(layer.buffers())),
                () -> assertEquals(List.of("scale", "bias"),
                        List.copyOf(parameterSnapshot.keySet())),
                () -> assertEquals(List.of("runningMean", "runningVariance"),
                        List.copyOf(bufferSnapshot.keySet())),
                () -> assertSame(layer.scale(), layer.parameters().get(0)),
                () -> assertSame(layer.bias(), layer.parameters().get(1)),
                () -> assertSame(layer.runningMean(), layer.buffers().get(0)),
                () -> assertSame(layer.runningVariance(), layer.buffers().get(1)),
                () -> assertSame(layer.scale(), parameterSnapshot.get("scale")),
                () -> assertSame(layer.bias(), parameterSnapshot.get("bias")),
                () -> assertSame(layer.runningMean(), bufferSnapshot.get("runningMean")),
                () -> assertSame(layer.runningVariance(), bufferSnapshot.get("runningVariance")),
                () -> assertSame(scale, layer.scale().value()),
                () -> assertSame(bias, layer.bias().value()),
                () -> assertSame(mean, layer.runningMean().value()),
                () -> assertSame(variance, layer.runningVariance().value()),
                () -> assertEquals(beforeConstruction, next.get()),
                () -> assertTrue(scale.descriptor().requiresGrad()),
                () -> assertTrue(bias.descriptor().requiresGrad()),
                () -> assertFalse(mean.descriptor().requiresGrad()),
                () -> assertFalse(variance.descriptor().requiresGrad()));

        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor output = layer.forward(input, new ForwardContext(ForwardMode.EVALUATION));
        BatchNormInferenceAttrs attrs = (BatchNormInferenceAttrs) output.provenance()
                .orElseThrow().operation().attrs();
        assertAll(
                () -> assertEquals(1, attrs.channelAxis()),
                () -> assertSame(epsilon, attrs.epsilon()),
                () -> assertSame(mean, bufferSnapshot.get("runningMean").value()),
                () -> assertSame(variance, bufferSnapshot.get("runningVariance").value()));
    }

    @Test
    void suppliedConstructionValidatesEveryStateStageAndScalarsWithoutTensorSideEffects()
            throws ReflectiveOperationException {
        Tensor validScale = tensor(DataType.FLOAT32, Shape.of(3), true);
        Tensor validBias = tensor(DataType.FLOAT32, Shape.of(3), true);
        Tensor validMean = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor validVariance = tensor(DataType.FLOAT32, Shape.of(3), false);
        ScalarValue momentum = ScalarValue.float32(0.25f);
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        assertAll(
                () -> assertTrue(constructionFailure(
                                -1, null, null, null, null, null, null)
                        .contains("channelAxis")),
                () -> assertEquals("scale", nullFailure(
                        0, null, null, null, null, null, null)),
                () -> assertEquals("bias", nullFailure(
                        0, validScale, null, null, null, null, null)),
                () -> assertEquals("runningMean", nullFailure(
                        0, validScale, validBias, null, null, null, null)),
                () -> assertEquals("runningVariance", nullFailure(
                        0, validScale, validBias, validMean, null, null, null)),
                () -> assertEquals("momentum", nullFailure(
                        0, validScale, validBias, validMean, validVariance, null, null)),
                () -> assertEquals("epsilon", nullFailure(
                        0, validScale, validBias, validMean, validVariance, momentum, null)),
                () -> assertContains("scale", "floating", tensor(DataType.INT32, Shape.of(3), false),
                        validBias, validMean, validVariance, momentum, epsilon),
                () -> assertContains("scale", "requiresGrad", tensor(DataType.FLOAT32, Shape.of(3), false),
                        validBias, validMean, validVariance, momentum, epsilon),
                () -> assertContains("scale", "rank one", tensor(DataType.FLOAT32, Shape.of(1, 3), true),
                        validBias, validMean, validVariance, momentum, epsilon),
                () -> assertContains("scale", "fully static", tensor(
                                DataType.FLOAT32,
                                Shape.ofDimensions(new DynamicDimension("C")),
                                true),
                        validBias, validMean, validVariance, momentum, epsilon),
                () -> assertContains("scale", "positive", tensor(DataType.FLOAT32, Shape.of(0), true),
                        tensor(DataType.FLOAT32, Shape.of(0), true),
                        tensor(DataType.FLOAT32, Shape.of(0), false),
                        tensor(DataType.FLOAT32, Shape.of(0), false), momentum, epsilon),
                () -> assertRoleFailure("bias", "floating", validScale,
                        tensor(DataType.INT64, Shape.of(3), false), validMean, validVariance,
                        momentum, epsilon),
                () -> assertRoleFailure("bias", "requiresGrad", validScale,
                        tensor(DataType.FLOAT32, Shape.of(3), false), validMean, validVariance,
                        momentum, epsilon),
                () -> assertRoleFailure("bias", "rank one", validScale,
                        tensor(DataType.FLOAT32, Shape.of(1, 3), true), validMean, validVariance,
                        momentum, epsilon),
                () -> assertRoleFailure("bias", "fully static", validScale,
                        tensor(DataType.FLOAT32,
                                Shape.ofDimensions(new DynamicDimension("C")), true),
                        validMean, validVariance, momentum, epsilon),
                () -> assertRoleFailure("bias", "data type", validScale,
                        tensor(DataType.FLOAT64, Shape.of(3), true), validMean, validVariance,
                        momentum, epsilon),
                () -> assertRoleFailure("bias", "Shape", validScale,
                        tensor(DataType.FLOAT32, Shape.of(4), true), validMean, validVariance,
                        momentum, epsilon),
                () -> assertRoleFailure("running mean", "floating", validScale, validBias,
                        tensor(DataType.BOOL, Shape.of(3), false), validVariance, momentum, epsilon),
                () -> assertRoleFailure("running mean", "requiresGrad", validScale, validBias,
                        tensor(DataType.FLOAT32, Shape.of(3), true), validVariance, momentum, epsilon),
                () -> assertRoleFailure("running mean", "rank one", validScale, validBias,
                        tensor(DataType.FLOAT32, Shape.of(1, 3), false), validVariance,
                        momentum, epsilon),
                () -> assertRoleFailure("running mean", "fully static", validScale, validBias,
                        tensor(DataType.FLOAT32,
                                Shape.ofDimensions(new DynamicDimension("C")), false),
                        validVariance, momentum, epsilon),
                () -> assertRoleFailure("running mean", "data type", validScale, validBias,
                        tensor(DataType.FLOAT64, Shape.of(3), false), validVariance,
                        momentum, epsilon),
                () -> assertRoleFailure("running mean", "Shape", validScale, validBias,
                        tensor(DataType.FLOAT32, Shape.of(4), false), validVariance,
                        momentum, epsilon),
                () -> assertRoleFailure("running variance", "floating", validScale, validBias,
                        validMean, tensor(DataType.INT32, Shape.of(3), false), momentum, epsilon),
                () -> assertRoleFailure("running variance", "requiresGrad", validScale, validBias,
                        validMean, tensor(DataType.FLOAT32, Shape.of(3), true), momentum, epsilon),
                () -> assertRoleFailure("running variance", "rank one", validScale, validBias,
                        validMean, tensor(DataType.FLOAT32, Shape.of(1, 3), false), momentum, epsilon),
                () -> assertRoleFailure("running variance", "fully static", validScale, validBias,
                        validMean, tensor(DataType.FLOAT32,
                                Shape.ofDimensions(new DynamicDimension("C")), false),
                        momentum, epsilon),
                () -> assertRoleFailure("running variance", "data type", validScale, validBias,
                        validMean, tensor(DataType.FLOAT64, Shape.of(3), false), momentum, epsilon),
                () -> assertRoleFailure("running variance", "Shape", validScale, validBias,
                        validMean, tensor(DataType.FLOAT32, Shape.of(4), false), momentum, epsilon),
                () -> assertTrue(constructionFailure(0, validScale, validBias, validMean,
                                validVariance, ScalarValue.float32(1.1f), epsilon)
                        .contains("momentum")),
                () -> assertTrue(constructionFailure(0, validScale, validBias, validMean,
                                validVariance, ScalarValue.int32(0), epsilon)
                        .contains("momentum must have a floating")),
                () -> assertTrue(constructionFailure(0, validScale, validBias, validMean,
                                validVariance, momentum, ScalarValue.float32(0.0f))
                        .contains("epsilon")),
                () -> assertTrue(constructionFailure(0, validScale, validBias, validMean,
                                validVariance, momentum, ScalarValue.int32(1))
                        .contains("epsilon must have a floating")),
                () -> assertTrue(constructionFailure(0, validScale, validBias, validMean,
                                validVariance, ScalarValue.float64(0.25),
                                ScalarValue.float64(1.0e-5))
                        .contains("momentum data type")),
                () -> assertTrue(constructionFailure(0, validScale, validBias, validMean,
                                validVariance, momentum, ScalarValue.float64(1.0e-5))
                        .contains("epsilon data type")));

        AtomicLong next = nextTensorIdState();
        long before = next.get();
        assertTrue(constructionFailure(0, validScale, validBias, validMean,
                        validVariance, momentum, ScalarValue.float64(1.0e-5))
                .contains("epsilon data type"));
        assertEquals(before, next.get());
    }

    @Test
    void evaluationUsesCapturedContextExactlyOnceAndNeverReplacesBuffers() {
        BatchNorm layer = layer(1);
        Tensor oldMean = layer.runningMean().value();
        Tensor oldVariance = layer.runningVariance().value();
        layer.eval();
        ForwardContext evaluationSnapshot = layer.forwardContext();
        layer.train();
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);

        Tensor first = layer.forward(input, evaluationSnapshot);
        Tensor second = layer.forward(input, new ForwardContext(ForwardMode.EVALUATION));
        BatchNorm firstAxisLayer = layer(0);
        Tensor firstAxisInput = tensor(DataType.FLOAT32, Shape.of(3, 2), false);
        Tensor firstAxisOutput = firstAxisLayer.forward(
                firstAxisInput, new ForwardContext(ForwardMode.EVALUATION));

        assertInference(first, input, layer.scale().value(), layer.bias().value(),
                oldMean, oldVariance, 1);
        assertInference(second, input, layer.scale().value(), layer.bias().value(),
                oldMean, oldVariance, 1);
        assertInference(
                firstAxisOutput,
                firstAxisInput,
                firstAxisLayer.scale().value(),
                firstAxisLayer.bias().value(),
                firstAxisLayer.runningMean().value(),
                firstAxisLayer.runningVariance().value(),
                0);
        assertAll(
                () -> assertEquals(ForwardMode.TRAINING, layer.mode()),
                () -> assertEquals(ForwardMode.EVALUATION, evaluationSnapshot.mode()),
                () -> assertSame(oldMean, layer.runningMean().value()),
                () -> assertSame(oldVariance, layer.runningVariance().value()),
                () -> assertNotSame(first, second),
                () -> assertNotSame(first.provenance().orElseThrow().producer(),
                        second.provenance().orElseThrow().producer()));
    }

    @Test
    void trainingUsesCapturedContextAndInstallsExactSharedProducerStatisticsInOrder() {
        ScalarValue momentum = ScalarValue.float32(0.25f);
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        BatchNorm layer = new BatchNorm(
                1,
                tensor(DataType.FLOAT32, Shape.of(3), true),
                tensor(DataType.FLOAT32, Shape.of(3), true),
                tensor(DataType.FLOAT32, Shape.of(3), false),
                tensor(DataType.FLOAT32, Shape.of(3), false),
                momentum,
                epsilon);
        Tensor scale = layer.scale().value();
        Tensor bias = layer.bias().value();
        Tensor oldMean = layer.runningMean().value();
        Tensor oldVariance = layer.runningVariance().value();
        Map<String, Buffer> structuralSnapshot = layer.buffersRecursively();
        ForwardContext trainingSnapshot = layer.forwardContext();
        layer.eval();
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3, 4), true);

        Tensor output = layer.forward(input, trainingSnapshot);
        TensorProducer producer = output.provenance().orElseThrow().producer();
        Tensor nextMean = producer.output(1);
        Tensor nextVariance = producer.output(2);

        assertTraining(
                output, input, scale, bias, oldMean, oldVariance, 1, momentum, epsilon);
        assertAll(
                () -> assertEquals(ForwardMode.EVALUATION, layer.mode()),
                () -> assertSame(nextMean, layer.runningMean().value()),
                () -> assertSame(nextVariance, layer.runningVariance().value()),
                () -> assertSame(nextMean, structuralSnapshot.get("runningMean").value()),
                () -> assertSame(nextVariance,
                        structuralSnapshot.get("runningVariance").value()),
                () -> assertSame(producer,
                        layer.runningMean().value().provenance().orElseThrow().producer()),
                () -> assertSame(producer,
                        layer.runningVariance().value().provenance().orElseThrow().producer()),
                () -> assertEquals(1,
                        layer.runningMean().value().provenance().orElseThrow().outputIndex()),
                () -> assertEquals(2,
                        layer.runningVariance().value().provenance().orElseThrow().outputIndex()),
                () -> assertSame(input.descriptor().shape(), output.descriptor().shape()),
                () -> assertTrue(output.descriptor().requiresGrad()),
                () -> assertTrue(nextMean.descriptor().requiresGrad()),
                () -> assertTrue(nextVariance.descriptor().requiresGrad()),
                () -> assertTrue(output.hostStorage().isEmpty()),
                () -> assertTrue(nextMean.hostStorage().isEmpty()),
                () -> assertTrue(nextVariance.hostStorage().isEmpty()));

        Tensor secondOutput = layer.forward(input, trainingSnapshot);
        TensorProducer secondProducer = secondOutput.provenance().orElseThrow().producer();
        assertAll(
                () -> assertNotSame(output, secondOutput),
                () -> assertNotEquals(output.id(), secondOutput.id()),
                () -> assertSame(nextMean, secondProducer.inputs().get(3)),
                () -> assertSame(nextVariance, secondProducer.inputs().get(4)),
                () -> assertSame(oldMean, producer.inputs().get(3)),
                () -> assertSame(oldVariance, producer.inputs().get(4)),
                () -> assertSame(secondProducer.output(1), layer.runningMean().value()),
                () -> assertSame(secondProducer.output(2), layer.runningVariance().value()));
    }

    @Test
    void compatibleParameterReplacementAffectsOnlyLaterForwardExpressions() {
        BatchNorm layer = layer(1);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        ForwardContext evaluation = new ForwardContext(ForwardMode.EVALUATION);
        Tensor oldScale = layer.scale().value();
        Tensor oldBias = layer.bias().value();
        Tensor before = layer.forward(input, evaluation);
        Tensor newScale = tensor(DataType.FLOAT32, Shape.of(3), true);
        Tensor newBias = tensor(DataType.FLOAT32, Shape.of(3), true);

        layer.scale().replace(newScale);
        layer.bias().replace(newBias);
        Tensor after = layer.forward(input, evaluation);

        assertAll(
                () -> assertSame(oldScale, before.provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(oldBias, before.provenance().orElseThrow().inputs().get(2)),
                () -> assertSame(newScale, after.provenance().orElseThrow().inputs().get(1)),
                () -> assertSame(newBias, after.provenance().orElseThrow().inputs().get(2)),
                () -> assertSame(newScale, layer.scale().value()),
                () -> assertSame(newBias, layer.bias().value()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> layer.scale().replace(
                                tensor(DataType.FLOAT32, Shape.of(4), true))),
                () -> assertSame(newScale, layer.scale().value()));
    }

    @Test
    void forwardValidationAndModelFailuresPreserveBothBuffers()
            throws ReflectiveOperationException {
        BatchNorm layer = layer(1);
        Tensor oldMean = layer.runningMean().value();
        Tensor oldVariance = layer.runningVariance().value();
        ForwardContext training = new ForwardContext(ForwardMode.TRAINING);

        assertAll(
                () -> assertEquals("input",
                        assertThrows(NullPointerException.class,
                                () -> layer.forward(null, null)).getMessage()),
                () -> assertEquals("context",
                        assertThrows(NullPointerException.class,
                                () -> layer.forward(
                                        tensor(DataType.FLOAT32, Shape.of(2, 3), false), null))
                                .getMessage()),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                                () -> layer.forward(
                                        tensor(DataType.FLOAT32, Shape.of(3), false), training))
                        .getMessage().contains("rank")),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                                () -> layer.forward(
                                        tensor(DataType.FLOAT32, Shape.of(2, 4), false), training))
                        .getMessage().contains("channel Dimension")),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                                () -> layer.forward(
                                        tensor(
                                                DataType.FLOAT32,
                                                Shape.ofDimensions(
                                                        new DynamicDimension("N"),
                                                        new DynamicDimension("C")),
                                                false),
                                        training))
                        .getMessage().contains("channel Dimension")),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                                () -> layer.forward(
                                        tensor(DataType.INT32, Shape.of(2, 3), false), training))
                        .getMessage().contains("floating")),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                                () -> layer.forward(
                                        tensor(DataType.FLOAT64, Shape.of(2, 3), false), training))
                        .getMessage().contains("momentum data type")),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                                () -> layer.forward(
                                        tensor(DataType.FLOAT32, Shape.of(1, 3), false), training))
                        .getMessage().contains("reduction domain")),
                () -> assertSame(oldMean, layer.runningMean().value()),
                () -> assertSame(oldVariance, layer.runningVariance().value()));

        BatchNorm axisLayer = layer(2);
        assertThrows(IndexOutOfBoundsException.class,
                () -> axisLayer.forward(
                        tensor(DataType.FLOAT32, Shape.of(2, 3), false), training));
        assertAll(
                () -> assertSame(oldMean, layer.runningMean().value()),
                () -> assertSame(oldVariance, layer.runningVariance().value()));
    }

    @Test
    void partialModelIdentifierFailureConsumesIdsButNeverTransitionsBuffers()
            throws ReflectiveOperationException {
        BatchNorm layer = layer(1);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor oldMean = layer.runningMean().value();
        Tensor oldVariance = layer.runningVariance().value();
        AtomicLong next = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long savedNext = next.get();
        boolean savedClaimed = maximumClaimed.get();
        try {
            next.set(Long.MAX_VALUE - 1);
            maximumClaimed.set(false);

            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> layer.forward(input, new ForwardContext(ForwardMode.TRAINING)));

            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(maximumClaimed.get()),
                    () -> assertSame(oldMean, layer.runningMean().value()),
                    () -> assertSame(oldVariance, layer.runningVariance().value()));
        } finally {
            next.set(savedNext);
            maximumClaimed.set(savedClaimed);
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void internalSecondReplacementFailureExposesDocumentedMeanThenVarianceNonTransaction()
            throws ReflectiveOperationException {
        BatchNorm layer = layer(1);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Buffer meanHandle = layer.runningMean();
        Buffer varianceHandle = layer.runningVariance();
        Tensor oldMean = meanHandle.value();
        Tensor oldVariance = varianceHandle.value();
        Field buffersField = Module.class.getDeclaredField("buffers");
        buffersField.setAccessible(true);
        Map<String, Buffer> buffers = (LinkedHashMap<String, Buffer>) buffersField.get(layer);
        buffers.remove("runningVariance");

        IllegalArgumentException failure = assertThrows(IllegalArgumentException.class,
                () -> layer.forward(input, new ForwardContext(ForwardMode.TRAINING)));
        Tensor installedMean = meanHandle.value();
        TensorProducer producer = installedMean.provenance().orElseThrow().producer();

        assertAll(
                () -> assertTrue(failure.getMessage().contains("runningVariance")),
                () -> assertNotSame(oldMean, installedMean),
                () -> assertSame(producer.output(1), installedMean),
                () -> assertSame(oldVariance, varianceHandle.value()),
                () -> assertSame(oldMean, producer.inputs().get(3)),
                () -> assertSame(oldVariance, producer.inputs().get(4)),
                () -> assertSame(BatchNormKind.BATCH_NORM_TRAINING,
                        producer.operation().kind()),
                () -> assertEquals(5, producer.outputCount()));
    }

    private static BatchNorm layer(int channelAxis) {
        return new BatchNorm(
                channelAxis,
                tensor(DataType.FLOAT32, Shape.of(3), true),
                tensor(DataType.FLOAT32, Shape.of(3), true),
                tensor(DataType.FLOAT32, Shape.of(3), false),
                tensor(DataType.FLOAT32, Shape.of(3), false),
                ScalarValue.float32(0.25f),
                ScalarValue.float32(1.0e-5f));
    }

    private static void assertInference(
            Tensor output,
            Tensor input,
            Tensor scale,
            Tensor bias,
            Tensor mean,
            Tensor variance,
            int channelAxis) {
        TensorProvenance provenance = output.provenance().orElseThrow();
        BatchNormInferenceAttrs attrs = (BatchNormInferenceAttrs) provenance.operation().attrs();
        assertAll(
                () -> assertSame(BatchNormKind.BATCH_NORM_INFERENCE,
                        provenance.operation().kind()),
                () -> assertEquals(channelAxis, attrs.channelAxis()),
                () -> assertEquals(List.of(input, scale, bias, mean, variance),
                        provenance.inputs()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertEquals(1, provenance.producer().outputCount()),
                () -> assertSame(output, provenance.producer().output(0)));
    }

    private static void assertTraining(
            Tensor output,
            Tensor input,
            Tensor scale,
            Tensor bias,
            Tensor mean,
            Tensor variance,
            int channelAxis,
            ScalarValue momentum,
            ScalarValue epsilon) {
        TensorProvenance provenance = output.provenance().orElseThrow();
        BatchNormTrainingAttrs attrs = (BatchNormTrainingAttrs) provenance.operation().attrs();
        assertAll(
                () -> assertSame(BatchNormKind.BATCH_NORM_TRAINING,
                        provenance.operation().kind()),
                () -> assertEquals(channelAxis, attrs.channelAxis()),
                () -> assertSame(momentum, attrs.momentum()),
                () -> assertSame(epsilon, attrs.epsilon()),
                () -> assertEquals(List.of(input, scale, bias, mean, variance),
                        provenance.inputs()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertEquals(5, provenance.producer().outputCount()),
                () -> assertSame(output, provenance.producer().output(0)));
    }

    private static void assertContains(
            String role,
            String expected,
            Tensor scale,
            Tensor bias,
            Tensor mean,
            Tensor variance,
            ScalarValue momentum,
            ScalarValue epsilon) {
        String failure = constructionFailure(
                0, scale, bias, mean, variance, momentum, epsilon);
        assertTrue(failure.contains(role) && failure.contains(expected), failure);
    }

    private static void assertRoleFailure(
            String role,
            String expected,
            Tensor scale,
            Tensor bias,
            Tensor mean,
            Tensor variance,
            ScalarValue momentum,
            ScalarValue epsilon) {
        assertContains(role, expected, scale, bias, mean, variance, momentum, epsilon);
    }

    private static String constructionFailure(
            int channelAxis,
            Tensor scale,
            Tensor bias,
            Tensor mean,
            Tensor variance,
            ScalarValue momentum,
            ScalarValue epsilon) {
        return assertThrows(IllegalArgumentException.class,
                () -> new BatchNorm(
                        channelAxis, scale, bias, mean, variance, momentum, epsilon))
                .getMessage();
    }

    private static String nullFailure(
            int channelAxis,
            Tensor scale,
            Tensor bias,
            Tensor mean,
            Tensor variance,
            ScalarValue momentum,
            ScalarValue epsilon) {
        return assertThrows(NullPointerException.class,
                () -> new BatchNorm(
                        channelAxis, scale, bias, mean, variance, momentum, epsilon))
                .getMessage();
    }

    private static List<String> names(List<Parameter> parameters) {
        return parameters.stream().map(Parameter::name).toList();
    }

    private static List<String> bufferNames(List<Buffer> buffers) {
        return buffers.stream().map(Buffer::name).toList();
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
