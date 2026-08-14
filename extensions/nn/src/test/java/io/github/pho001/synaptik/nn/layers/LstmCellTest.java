package io.github.pho001.synaptik.nn.layers;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProvenance;
import io.github.pho001.synaptik.nn.module.Buffer;
import io.github.pho001.synaptik.nn.module.ForwardMode;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import io.github.pho001.synaptik.nn.module.Sequential;
import io.github.pho001.synaptik.nn.module.StateKind;
import io.github.pho001.synaptik.nn.module.UnaryTensorModule;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.random.RandomGenerator;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.parallel.Execution;
import org.junit.jupiter.api.parallel.ExecutionMode;

@Execution(ExecutionMode.SAME_THREAD)
class LstmCellTest {
    @Test
    void exposesExactlyThePlannedFinalDirectModuleAndResultSurfaces() throws Exception {
        Set<List<Class<?>>> constructors = Arrays.stream(LstmCell.class.getDeclaredConstructors())
                .filter(constructor -> Modifier.isPublic(constructor.getModifiers()))
                .map(Constructor::getParameterTypes)
                .map(List::of)
                .collect(Collectors.toSet());
        Set<String> methods = Arrays.stream(LstmCell.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(Method::getName)
                .collect(Collectors.toSet());
        Set<String> resultMethods = Arrays.stream(
                        LstmCellForwardResult.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(Method::getName)
                .collect(Collectors.toSet());

        assertAll(
                () -> assertTrue(Modifier.isPublic(LstmCell.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(LstmCell.class.getModifiers())),
                () -> assertSame(Module.class, LstmCell.class.getSuperclass()),
                () -> assertFalse(UnaryTensorModule.class.isAssignableFrom(LstmCell.class)),
                () -> assertFalse(Sequential.class.isAssignableFrom(LstmCell.class)),
                () -> assertEquals(3, LstmCell.class.getDeclaredConstructors().length),
                () -> assertTrue(Arrays.stream(LstmCell.class.getDeclaredConstructors())
                        .allMatch(constructor -> Modifier.isPublic(constructor.getModifiers()))),
                () -> assertEquals(Set.of(
                        List.of(Tensor.class, Tensor.class),
                        List.of(Tensor.class, Tensor.class, Tensor.class),
                        List.of(long.class, long.class, boolean.class,
                                DataType.class, RandomGenerator.class)), constructors),
                () -> assertEquals(Set.of("inputWeight", "hiddenWeight", "bias", "forward"),
                        methods),
                () -> assertSame(LstmCellForwardResult.class,
                        LstmCell.class.getDeclaredMethod(
                                "forward", Tensor.class, Tensor.class, Tensor.class)
                                .getReturnType()),
                () -> assertTrue(Arrays.stream(LstmCell.class.getDeclaredMethods())
                        .noneMatch(method -> Modifier.isProtected(method.getModifiers()))),
                () -> assertEquals(Set.of("inputWeight", "hiddenWeight", "bias"),
                        Arrays.stream(LstmCell.class.getDeclaredFields())
                                .map(Field::getName).collect(Collectors.toSet())),
                () -> assertTrue(Arrays.stream(LstmCell.class.getDeclaredFields())
                        .allMatch(field -> Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertTrue(Arrays.stream(LstmCell.class.getDeclaredFields())
                        .noneMatch(field -> field.getType() == Tensor.class
                                || field.getType() == Buffer.class
                                || RandomGenerator.class.isAssignableFrom(field.getType()))),
                () -> assertEquals(0, LstmCell.class.getDeclaredClasses().length),
                () -> assertTrue(LstmCellForwardResult.class.isRecord()),
                () -> assertTrue(Modifier.isPublic(LstmCellForwardResult.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(LstmCellForwardResult.class.getModifiers())),
                () -> assertEquals(1,
                        LstmCellForwardResult.class.getDeclaredConstructors().length),
                () -> assertEquals(
                        List.of(Tensor.class, Tensor.class),
                        List.of(LstmCellForwardResult.class.getDeclaredConstructors()[0]
                                .getParameterTypes())),
                () -> assertEquals(List.of("nextHidden", "nextCell"),
                        Arrays.stream(LstmCellForwardResult.class.getRecordComponents())
                                .map(component -> component.getName()).toList()),
                () -> assertEquals(
                        Set.of("nextHidden", "nextCell", "equals", "hashCode", "toString"),
                        resultMethods),
                () -> assertTrue(Arrays.stream(LstmCellForwardResult.class.getDeclaredMethods())
                        .noneMatch(method -> Modifier.isProtected(method.getModifiers()))),
                () -> assertEquals(2,
                        LstmCellForwardResult.class.getDeclaredFields().length),
                () -> assertTrue(Arrays.stream(
                                LstmCellForwardResult.class.getDeclaredFields())
                        .allMatch(field -> field.getType() == Tensor.class
                                && Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(0, LstmCellForwardResult.class.getDeclaredClasses().length));
    }

    @Test
    void resultRejectsNullsInOrderRetainsExactReferencesAndAllocatesNoTensorId()
            throws Exception {
        Tensor nextHidden = tensor(DataType.FLOAT32, Shape.of(4), false);
        Tensor nextCell = tensor(DataType.FLOAT32, Shape.of(4), false);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertEquals("nextHidden", assertThrows(
                        NullPointerException.class,
                        () -> new LstmCellForwardResult(null, null)).getMessage()),
                () -> assertEquals("nextCell", assertThrows(
                        NullPointerException.class,
                        () -> new LstmCellForwardResult(nextHidden, null)).getMessage()),
                () -> {
                    LstmCellForwardResult result =
                            new LstmCellForwardResult(nextHidden, nextCell);
                    assertSame(nextHidden, result.nextHidden());
                    assertSame(nextCell, result.nextCell());
                    assertEquals(result,
                            new LstmCellForwardResult(nextHidden, nextCell));
                },
                () -> assertEquals(before, ids.get()));
    }

    @Test
    void suppliedStateRetainsPackedBindingsNamesOrderAndDiscovery() {
        Tensor inputWeight = tensor(DataType.FLOAT32, Shape.of(16, 3), true);
        Tensor hiddenWeight = tensor(DataType.FLOAT32, Shape.of(16, 4), true);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(16), true);
        LstmCell noBias = new LstmCell(inputWeight, hiddenWeight);
        LstmCell biased = new LstmCell(inputWeight, hiddenWeight, bias);
        Owner owner = new Owner(biased);

        assertAll(
                () -> assertSame(inputWeight, noBias.inputWeight().value()),
                () -> assertSame(hiddenWeight, noBias.hiddenWeight().value()),
                () -> assertTrue(noBias.bias().isEmpty()),
                () -> assertEquals(List.of("inputWeight", "hiddenWeight"), names(noBias)),
                () -> assertSame(inputWeight, biased.inputWeight().value()),
                () -> assertSame(hiddenWeight, biased.hiddenWeight().value()),
                () -> assertSame(bias, biased.bias().orElseThrow().value()),
                () -> assertEquals(List.of("inputWeight", "hiddenWeight", "bias"), names(biased)),
                () -> assertTrue(biased.buffers().isEmpty()),
                () -> assertTrue(biased.children().isEmpty()),
                () -> assertEquals(
                        List.of("cell.inputWeight", "cell.hiddenWeight", "cell.bias"),
                        List.copyOf(owner.parametersRecursively().keySet())),
                () -> assertTrue(owner.stateDictionary().entries().stream()
                        .allMatch(entry -> entry.kind() == StateKind.PARAMETER)),
                () -> assertEquals(
                        List.of("cell.inputWeight", "cell.hiddenWeight", "cell.bias"),
                        owner.stateDictionary().entries().stream().map(entry -> entry.path()).toList()),
                () -> assertTrue(owner.buffersRecursively().isEmpty()));
    }

    @Test
    void suppliedConstructionValidatesCompletePackedSchemaBeforeDeclarationOrIds()
            throws Exception {
        Tensor validInput = tensor(DataType.FLOAT32, Shape.of(16, 3), true);
        Tensor validHidden = tensor(DataType.FLOAT32, Shape.of(16, 4), true);
        Tensor integral = tensor(DataType.INT32, Shape.of(16, 3), false);
        Tensor noGradient = tensor(DataType.FLOAT32, Shape.of(16, 3), false);
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), true);
        Tensor dynamic = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("P"), new StaticDimension(3)), true);
        Tensor zeroPacked = tensor(DataType.FLOAT32, Shape.of(0, 3), true);
        Tensor unpacked = tensor(DataType.FLOAT32, Shape.of(14, 3), true);
        Tensor zeroInput = tensor(DataType.FLOAT32, Shape.of(16, 0), true);
        assertHiddenAndBiasFailures(validInput, validHidden);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertEquals("inputWeight", assertThrows(NullPointerException.class,
                        () -> new LstmCell(null, null, null)).getMessage()),
                () -> assertEquals("hiddenWeight", assertThrows(NullPointerException.class,
                        () -> new LstmCell(validInput, null, null)).getMessage()),
                () -> assertEquals("bias", assertThrows(NullPointerException.class,
                        () -> new LstmCell(validInput, validHidden, null)).getMessage()),
                () -> assertContains("inputWeight", "floating",
                        () -> new LstmCell(integral, validHidden)),
                () -> assertContains("inputWeight", "requiresGrad",
                        () -> new LstmCell(noGradient, validHidden)),
                () -> assertContains("inputWeight", "rank two",
                        () -> new LstmCell(scalar, validHidden)),
                () -> assertContains("inputWeight", "fully static",
                        () -> new LstmCell(dynamic, validHidden)),
                () -> assertContains("packed hidden size", "positive",
                        () -> new LstmCell(zeroPacked, validHidden)),
                () -> assertContains("packed hidden size", "divisible",
                        () -> new LstmCell(unpacked, validHidden)),
                () -> assertContains("inputSize", "positive",
                        () -> new LstmCell(zeroInput, validHidden)),
                () -> assertEquals(before, ids.get()));
    }

    @Test
    void forwardBuildsExactFourGateFormulaBiasSideResultReferencesAndIdentifierOrder()
            throws Exception {
        Tensor inputWeight = tensor(DataType.FLOAT32, Shape.of(16, 3), true);
        Tensor hiddenWeight = tensor(DataType.FLOAT32, Shape.of(16, 4), true);
        Tensor bias = tensor(DataType.FLOAT32, Shape.of(16), true);
        Tensor input = tensor(DataType.FLOAT64, Shape.of(2, 3), false);
        Tensor hidden = tensor(DataType.BFLOAT16, Shape.of(2, 4), false);
        Tensor cell = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        AtomicLong ids = nextTensorIdState();

        long noBiasStart = ids.get();
        LstmCellForwardResult noBias =
                new LstmCell(inputWeight, hiddenWeight).forward(input, hidden, cell);
        assertLstmChain(noBias, input, hidden, cell,
                inputWeight, hiddenWeight, Optional.empty());
        assertIds(noBias, noBiasStart, false);

        long biasedStart = ids.get();
        LstmCellForwardResult biased =
                new LstmCell(inputWeight, hiddenWeight, bias).forward(input, hidden, cell);
        assertLstmChain(biased, input, hidden, cell,
                inputWeight, hiddenWeight, Optional.of(bias));
        assertIds(biased, biasedStart, true);
        assertAll(
                () -> assertSame(DataType.FLOAT64,
                        biased.nextHidden().descriptor().dataType()),
                () -> assertSame(DataType.FLOAT64,
                        biased.nextCell().descriptor().dataType()),
                () -> assertEquals(Shape.of(2, 4),
                        biased.nextHidden().descriptor().shape()),
                () -> assertEquals(Shape.of(2, 4),
                        biased.nextCell().descriptor().shape()),
                () -> assertTrue(biased.nextHidden().descriptor().requiresGrad()),
                () -> assertTrue(biased.nextCell().descriptor().requiresGrad()));
    }

    @Test
    void forwardSupportsRankOneMixedFloatingAndIndependentLeadingBroadcasting() {
        LstmCell lstm = cell(DataType.FLOAT32, true);
        DynamicDimension batch = new DynamicDimension("B");

        LstmCellForwardResult rankOne = lstm.forward(
                tensor(DataType.BFLOAT16, Shape.of(3), false),
                tensor(DataType.FLOAT32, Shape.of(4), false),
                tensor(DataType.FLOAT64, Shape.of(4), false));
        LstmCellForwardResult stateVectorsBroadcast = lstm.forward(
                tensor(DataType.FLOAT64, Shape.of(7, 3), false),
                tensor(DataType.BFLOAT16, Shape.of(4), false),
                tensor(DataType.FLOAT32, Shape.of(4), false));
        LstmCellForwardResult singletonBroadcast = lstm.forward(
                tensor(DataType.FLOAT32, Shape.of(2, 1, 3), false),
                tensor(DataType.FLOAT32, Shape.of(7, 4), false),
                tensor(DataType.FLOAT32, Shape.of(1, 7, 4), false));
        LstmCellForwardResult symbolic = lstm.forward(
                tensor(DataType.FLOAT32,
                        Shape.ofDimensions(batch, new StaticDimension(3)), false),
                tensor(DataType.FLOAT32,
                        Shape.ofDimensions(batch, new StaticDimension(4)), false),
                tensor(DataType.FLOAT32,
                        Shape.ofDimensions(batch, new StaticDimension(4)), false));

        assertAll(
                () -> assertEquals(Shape.of(4), rankOne.nextHidden().descriptor().shape()),
                () -> assertEquals(Shape.of(4), rankOne.nextCell().descriptor().shape()),
                () -> assertSame(DataType.FLOAT64,
                        rankOne.nextHidden().descriptor().dataType()),
                () -> assertEquals(Shape.of(7, 4),
                        stateVectorsBroadcast.nextHidden().descriptor().shape()),
                () -> assertEquals(Shape.of(2, 7, 4),
                        singletonBroadcast.nextCell().descriptor().shape()),
                () -> assertEquals(
                        Shape.ofDimensions(batch, new StaticDimension(4)),
                        symbolic.nextHidden().descriptor().shape()));
    }

    @Test
    void forwardPrevalidatesEveryLocallyKnowableFailureBeforeFirstExpressionId()
            throws Exception {
        LstmCell lstm = cell(DataType.FLOAT32, true);
        Tensor validInput = tensor(DataType.FLOAT32, Shape.of(2, 3), false);
        Tensor validHidden = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor validCell = tensor(DataType.FLOAT32, Shape.of(2, 4), false);
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), false);
        Tensor integralInput = tensor(DataType.INT32, Shape.of(2, 3), false);
        Tensor integralHidden = tensor(DataType.INT64, Shape.of(2, 4), false);
        Tensor integralCell = tensor(DataType.INT32, Shape.of(2, 4), false);
        Tensor wrongInput = tensor(DataType.FLOAT32, Shape.of(2, 5), false);
        Tensor wrongHidden = tensor(DataType.FLOAT32, Shape.of(2, 5), false);
        Tensor wrongCell = tensor(DataType.FLOAT32, Shape.of(2, 5), false);
        Tensor incompatibleHidden = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        Tensor incompatibleCell = tensor(DataType.FLOAT32, Shape.of(3, 4), false);
        DynamicDimension left = new DynamicDimension("L");
        DynamicDimension right = new DynamicDimension("R");
        Tensor symbolicInput = tensor(DataType.FLOAT32,
                Shape.ofDimensions(left, new StaticDimension(3)), false);
        Tensor symbolicHidden = tensor(DataType.FLOAT32,
                Shape.ofDimensions(right, new StaticDimension(4)), false);
        Tensor unresolvedCellFeature = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("H")), false);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertEquals("input", assertThrows(NullPointerException.class,
                        () -> lstm.forward(null, null, null)).getMessage()),
                () -> assertEquals("hidden", assertThrows(NullPointerException.class,
                        () -> lstm.forward(validInput, null, null)).getMessage()),
                () -> assertEquals("cell", assertThrows(NullPointerException.class,
                        () -> lstm.forward(validInput, validHidden, null)).getMessage()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> lstm.forward(integralInput, validHidden, validCell)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> lstm.forward(validInput, integralHidden, validCell)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> lstm.forward(validInput, validHidden, integralCell)),
                () -> assertContains("input", "rank",
                        () -> lstm.forward(scalar, validHidden, validCell)),
                () -> assertContains("hidden", "rank",
                        () -> lstm.forward(validInput, scalar, validCell)),
                () -> assertContains("cell", "rank",
                        () -> lstm.forward(validInput, validHidden, scalar)),
                () -> assertContains("input", "feature",
                        () -> lstm.forward(wrongInput, validHidden, validCell)),
                () -> assertContains("hidden", "feature",
                        () -> lstm.forward(validInput, wrongHidden, validCell)),
                () -> assertContains("cell", "feature",
                        () -> lstm.forward(validInput, validHidden, wrongCell)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> lstm.forward(validInput, incompatibleHidden, validCell)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> lstm.forward(validInput, validHidden, incompatibleCell)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> lstm.forward(symbolicInput, symbolicHidden, validCell)),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> lstm.forward(validInput, validHidden, unresolvedCellFeature)),
                () -> assertEquals(before, ids.get()));
    }

    @Test
    void modeAndRepeatedExplicitTwoStateThreadingNeverCreateRetainedState() {
        LstmCell lstm = cell(DataType.FLOAT32, false);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor initialHidden = tensor(DataType.FLOAT32, Shape.of(4), false);
        Tensor initialCell = tensor(DataType.FLOAT32, Shape.of(4), false);

        lstm.eval();
        LstmCellForwardResult evaluation = lstm.forward(input, initialHidden, initialCell);
        lstm.train();
        LstmCellForwardResult training = lstm.forward(input, initialHidden, initialCell);
        LstmCellForwardResult threaded =
                lstm.forward(input, evaluation.nextHidden(), evaluation.nextCell());

        assertAll(
                () -> assertEquals(ForwardMode.TRAINING, lstm.mode()),
                () -> assertEquals(evaluation.nextHidden().descriptor(),
                        training.nextHidden().descriptor()),
                () -> assertNotSame(evaluation, training),
                () -> assertNotSame(training, threaded),
                () -> assertSame(initialCell, forgetProduct(training).inputs().get(1)),
                () -> assertSame(evaluation.nextCell(),
                        forgetProduct(threaded).inputs().get(1)),
                () -> assertSame(evaluation.nextHidden(),
                        hiddenProjection(threaded).provenance().orElseThrow().inputs().getFirst()),
                () -> assertTrue(lstm.buffersRecursively().isEmpty()),
                () -> assertEquals(List.of("inputWeight", "hiddenWeight"), names(lstm)),
                () -> assertEquals(2, lstm.stateDictionary().entries().size()));
    }

    @Test
    void compatibleReplacementAffectsOnlyLaterSnapshotsAndKeepsStableWrappers() {
        Tensor oldInputWeight = tensor(DataType.FLOAT32, Shape.of(16, 3), true);
        Tensor oldHiddenWeight = tensor(DataType.FLOAT32, Shape.of(16, 4), true);
        Tensor oldBias = tensor(DataType.FLOAT32, Shape.of(16), true);
        Tensor input = tensor(DataType.FLOAT32, Shape.of(3), false);
        Tensor hidden = tensor(DataType.FLOAT32, Shape.of(4), false);
        Tensor cellState = tensor(DataType.FLOAT32, Shape.of(4), false);
        LstmCell lstm = new LstmCell(oldInputWeight, oldHiddenWeight, oldBias);
        Parameter inputHandle = lstm.inputWeight();
        Parameter hiddenHandle = lstm.hiddenWeight();
        Parameter biasHandle = lstm.bias().orElseThrow();
        LstmCellForwardResult before = lstm.forward(input, hidden, cellState);

        Tensor newInputWeight = tensor(DataType.FLOAT32, Shape.of(16, 3), true);
        Tensor newHiddenWeight = tensor(DataType.FLOAT32, Shape.of(16, 4), true);
        Tensor newBias = tensor(DataType.FLOAT32, Shape.of(16), true);
        inputHandle.replace(newInputWeight);
        hiddenHandle.replace(newHiddenWeight);
        biasHandle.replace(newBias);
        LstmCellForwardResult after = lstm.forward(input, hidden, cellState);

        assertLstmChain(before, input, hidden, cellState,
                oldInputWeight, oldHiddenWeight, Optional.of(oldBias));
        assertLstmChain(after, input, hidden, cellState,
                newInputWeight, newHiddenWeight, Optional.of(newBias));
        assertAll(
                () -> assertSame(inputHandle, lstm.inputWeight()),
                () -> assertSame(hiddenHandle, lstm.hiddenWeight()),
                () -> assertSame(biasHandle, lstm.bias().orElseThrow()),
                () -> assertSame(newInputWeight, inputHandle.value()),
                () -> assertSame(newHiddenWeight, hiddenHandle.value()),
                () -> assertSame(newBias, biasHandle.value()),
                () -> assertThrows(IllegalArgumentException.class,
                        () -> hiddenHandle.replace(
                                tensor(DataType.FLOAT32, Shape.of(16, 5), true))),
                () -> assertSame(newHiddenWeight, hiddenHandle.value()));
    }

    private static void assertHiddenAndBiasFailures(Tensor inputWeight, Tensor validHidden)
            throws Exception {
        Tensor integralHidden = tensor(DataType.INT32, Shape.of(16, 4), false);
        Tensor noGradientHidden = tensor(DataType.FLOAT32, Shape.of(16, 4), false);
        Tensor scalar = tensor(DataType.FLOAT32, Shape.scalar(), true);
        Tensor dynamicHidden = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("P"), new StaticDimension(4)), true);
        Tensor zeroAxisZero = tensor(DataType.FLOAT32, Shape.of(0, 4), true);
        Tensor zeroAxisOne = tensor(DataType.FLOAT32, Shape.of(16, 0), true);
        Tensor wrongTypeHidden = tensor(DataType.FLOAT64, Shape.of(16, 4), true);
        Tensor wrongPacked = tensor(DataType.FLOAT32, Shape.of(20, 4), true);
        Tensor wrongHiddenSize = tensor(DataType.FLOAT32, Shape.of(16, 5), true);
        Tensor integralBias = tensor(DataType.INT32, Shape.of(16), false);
        Tensor noGradientBias = tensor(DataType.FLOAT32, Shape.of(16), false);
        Tensor dynamicBias = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("P")), true);
        Tensor wrongTypeBias = tensor(DataType.FLOAT64, Shape.of(16), true);
        Tensor wrongBias = tensor(DataType.FLOAT32, Shape.of(15), true);
        AtomicLong ids = nextTensorIdState();
        long before = ids.get();

        assertAll(
                () -> assertContains("hiddenWeight", "floating",
                        () -> new LstmCell(inputWeight, integralHidden)),
                () -> assertContains("hiddenWeight", "requiresGrad",
                        () -> new LstmCell(inputWeight, noGradientHidden)),
                () -> assertContains("hiddenWeight", "rank two",
                        () -> new LstmCell(inputWeight, scalar)),
                () -> assertContains("hiddenWeight", "fully static",
                        () -> new LstmCell(inputWeight, dynamicHidden)),
                () -> assertContains("axis zero", "positive",
                        () -> new LstmCell(inputWeight, zeroAxisZero)),
                () -> assertContains("axis one", "positive",
                        () -> new LstmCell(inputWeight, zeroAxisOne)),
                () -> assertContains("hiddenWeight", "data type",
                        () -> new LstmCell(inputWeight, wrongTypeHidden)),
                () -> assertContains("axis zero", "packed extent",
                        () -> new LstmCell(inputWeight, wrongPacked)),
                () -> assertContains("axis one", "hidden size",
                        () -> new LstmCell(inputWeight, wrongHiddenSize)),
                () -> assertContains("bias", "floating",
                        () -> new LstmCell(inputWeight, validHidden, integralBias)),
                () -> assertContains("bias", "requiresGrad",
                        () -> new LstmCell(inputWeight, validHidden, noGradientBias)),
                () -> assertContains("bias", "rank one",
                        () -> new LstmCell(inputWeight, validHidden, scalar)),
                () -> assertContains("bias", "fully static",
                        () -> new LstmCell(inputWeight, validHidden, dynamicBias)),
                () -> assertContains("bias", "data type",
                        () -> new LstmCell(inputWeight, validHidden, wrongTypeBias)),
                () -> assertContains("bias", "packed hidden size",
                        () -> new LstmCell(inputWeight, validHidden, wrongBias)),
                () -> assertEquals(before, ids.get()));
    }

    private static void assertLstmChain(
            LstmCellForwardResult result,
            Tensor input,
            Tensor hidden,
            Tensor cell,
            Tensor inputWeight,
            Tensor hiddenWeight,
            Optional<Tensor> bias) {
        Tensor nextHidden = result.nextHidden();
        Tensor nextCell = result.nextCell();
        TensorProvenance hiddenMul = nextHidden.provenance().orElseThrow();
        Tensor outputGate = hiddenMul.inputs().getFirst();
        Tensor activatedCell = hiddenMul.inputs().get(1);
        TensorProvenance activatedCellTanh = activatedCell.provenance().orElseThrow();
        TensorProvenance cellAdd = nextCell.provenance().orElseThrow();
        Tensor retainedCell = cellAdd.inputs().getFirst();
        Tensor selectedCandidate = cellAdd.inputs().get(1);
        TensorProvenance retainedCellMul = retainedCell.provenance().orElseThrow();
        Tensor forgetGate = retainedCellMul.inputs().getFirst();
        TensorProvenance selectedCandidateMul = selectedCandidate.provenance().orElseThrow();
        Tensor inputGate = selectedCandidateMul.inputs().getFirst();
        Tensor candidate = selectedCandidateMul.inputs().get(1);
        Tensor inputAdd = activationInput(inputGate);
        Tensor forgetAdd = activationInput(forgetGate);
        Tensor candidateAdd = activationInput(candidate);
        Tensor outputAdd = activationInput(outputGate);
        Tensor inputGateProjection = inputAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenInputGate = inputAdd.provenance().orElseThrow().inputs().get(1);
        Tensor forgetGateProjection = forgetAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenForgetGate = forgetAdd.provenance().orElseThrow().inputs().get(1);
        Tensor inputCandidate = candidateAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenCandidate = candidateAdd.provenance().orElseThrow().inputs().get(1);
        Tensor outputGateProjection = outputAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenOutputGate = outputAdd.provenance().orElseThrow().inputs().get(1);
        Tensor inputProjection =
                inputGateProjection.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenProjection =
                hiddenInputGate.provenance().orElseThrow().inputs().getFirst();

        assertAll(
                () -> assertSame(BinaryArithmeticKind.MUL, hiddenMul.operation().kind()),
                () -> assertSame(UnaryElementwiseKind.TANH,
                        activatedCellTanh.operation().kind()),
                () -> assertSame(nextCell, activatedCellTanh.inputs().getFirst()),
                () -> assertSame(BinaryArithmeticKind.ADD, cellAdd.operation().kind()),
                () -> assertSame(BinaryArithmeticKind.MUL,
                        retainedCellMul.operation().kind()),
                () -> assertSame(cell, retainedCellMul.inputs().get(1)),
                () -> assertSame(BinaryArithmeticKind.MUL,
                        selectedCandidateMul.operation().kind()),
                () -> assertSame(UnaryElementwiseKind.SIGMOID,
                        inputGate.provenance().orElseThrow().operation().kind()),
                () -> assertSame(UnaryElementwiseKind.SIGMOID,
                        forgetGate.provenance().orElseThrow().operation().kind()),
                () -> assertSame(UnaryElementwiseKind.TANH,
                        candidate.provenance().orElseThrow().operation().kind()),
                () -> assertSame(UnaryElementwiseKind.SIGMOID,
                        outputGate.provenance().orElseThrow().operation().kind()),
                () -> assertSlice(inputGateProjection, inputProjection, 0, 4),
                () -> assertSlice(forgetGateProjection, inputProjection, 4, 4),
                () -> assertSlice(inputCandidate, inputProjection, 8, 4),
                () -> assertSlice(outputGateProjection, inputProjection, 12, 4),
                () -> assertSlice(hiddenInputGate, hiddenProjection, 0, 4),
                () -> assertSlice(hiddenForgetGate, hiddenProjection, 4, 4),
                () -> assertSlice(hiddenCandidate, hiddenProjection, 8, 4),
                () -> assertSlice(hiddenOutputGate, hiddenProjection, 12, 4),
                () -> assertLinearChain(inputProjection, input, inputWeight, bias),
                () -> assertLinearChain(hiddenProjection, hidden, hiddenWeight, Optional.empty()));
    }

    private static Tensor activationInput(Tensor activation) {
        return activation.provenance().orElseThrow().inputs().getFirst();
    }

    private static void assertSlice(Tensor slice, Tensor projection, long start, long length) {
        TensorProvenance provenance = slice.provenance().orElseThrow();
        SliceAttrs attrs = (SliceAttrs) provenance.operation().attrs();
        int finalAxis = projection.descriptor().shape().rank() - 1;
        assertAll(
                () -> assertSame(SliceKind.SLICE, provenance.operation().kind()),
                () -> assertSame(projection, provenance.inputs().getFirst()),
                () -> assertEquals(List.of(start), attrs.starts()),
                () -> assertEquals(List.of(length), attrs.lengths()),
                () -> assertEquals(List.of(finalAxis), attrs.axes()),
                () -> assertEquals(List.of(1L), attrs.steps()));
    }

    private static void assertLinearChain(
            Tensor projection, Tensor input, Tensor weight, Optional<Tensor> bias) {
        Tensor product;
        if (bias.isPresent()) {
            TensorProvenance biasAdd = projection.provenance().orElseThrow();
            product = biasAdd.inputs().getFirst();
            assertAll(
                    () -> assertSame(BinaryArithmeticKind.ADD, biasAdd.operation().kind()),
                    () -> assertSame(bias.orElseThrow(), biasAdd.inputs().get(1)));
        } else {
            product = projection;
        }
        TensorProvenance matmul = product.provenance().orElseThrow();
        Tensor transposedWeight = matmul.inputs().get(1);
        TensorProvenance permute = transposedWeight.provenance().orElseThrow();
        assertAll(
                () -> assertSame(MatmulKind.MATMUL, matmul.operation().kind()),
                () -> assertSame(input, matmul.inputs().getFirst()),
                () -> assertSame(AxisTransformKind.PERMUTE, permute.operation().kind()),
                () -> assertEquals(List.of(1, 0),
                        ((PermutationAttrs) permute.operation().attrs()).axes()),
                () -> assertSame(weight, permute.inputs().getFirst()));
    }

    private static void assertIds(LstmCellForwardResult result, long start, boolean biased) {
        List<Tensor> ordered = orderedCreatedTensors(result, biased);
        assertEquals(biased ? 26 : 25, ordered.size());
        for (int index = 0; index < ordered.size(); index++) {
            assertEquals(start + index, ordered.get(index).id().value(), "ID at " + index);
        }
    }

    private static List<Tensor> orderedCreatedTensors(
            LstmCellForwardResult result, boolean biased) {
        Tensor nextHidden = result.nextHidden();
        Tensor nextCell = result.nextCell();
        Tensor outputGate = nextHidden.provenance().orElseThrow().inputs().getFirst();
        Tensor activatedCell = nextHidden.provenance().orElseThrow().inputs().get(1);
        Tensor retainedCell = nextCell.provenance().orElseThrow().inputs().getFirst();
        Tensor selectedCandidate = nextCell.provenance().orElseThrow().inputs().get(1);
        Tensor forgetGate = retainedCell.provenance().orElseThrow().inputs().getFirst();
        Tensor inputGate = selectedCandidate.provenance().orElseThrow().inputs().getFirst();
        Tensor candidate = selectedCandidate.provenance().orElseThrow().inputs().get(1);
        Tensor inputAdd = activationInput(inputGate);
        Tensor forgetAdd = activationInput(forgetGate);
        Tensor candidateAdd = activationInput(candidate);
        Tensor outputAdd = activationInput(outputGate);
        Tensor inputGateProjection = inputAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenInputGate = inputAdd.provenance().orElseThrow().inputs().get(1);
        Tensor forgetGateProjection = forgetAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenForgetGate = forgetAdd.provenance().orElseThrow().inputs().get(1);
        Tensor inputCandidate = candidateAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenCandidate = candidateAdd.provenance().orElseThrow().inputs().get(1);
        Tensor outputGateProjection = outputAdd.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenOutputGate = outputAdd.provenance().orElseThrow().inputs().get(1);
        Tensor inputProjection =
                inputGateProjection.provenance().orElseThrow().inputs().getFirst();
        Tensor hiddenProjection =
                hiddenInputGate.provenance().orElseThrow().inputs().getFirst();
        Tensor inputProduct = biased
                ? inputProjection.provenance().orElseThrow().inputs().getFirst()
                : inputProjection;
        Tensor inputTranspose = inputProduct.provenance().orElseThrow().inputs().get(1);
        Tensor hiddenTranspose = hiddenProjection.provenance().orElseThrow().inputs().get(1);

        ArrayList<Tensor> tensors = new ArrayList<>();
        tensors.add(inputTranspose);
        tensors.add(inputProduct);
        if (biased) {
            tensors.add(inputProjection);
        }
        tensors.add(hiddenTranspose);
        tensors.add(hiddenProjection);
        tensors.addAll(List.of(
                inputGateProjection, forgetGateProjection, inputCandidate, outputGateProjection,
                hiddenInputGate, hiddenForgetGate, hiddenCandidate, hiddenOutputGate,
                inputAdd, inputGate, forgetAdd, forgetGate, candidateAdd, candidate,
                outputAdd, outputGate, retainedCell, selectedCandidate, nextCell,
                activatedCell, nextHidden));
        return tensors;
    }

    private static TensorProvenance forgetProduct(LstmCellForwardResult result) {
        Tensor retainedCell = result.nextCell().provenance().orElseThrow().inputs().getFirst();
        return retainedCell.provenance().orElseThrow();
    }

    private static Tensor hiddenProjection(LstmCellForwardResult result) {
        Tensor outputGate = result.nextHidden().provenance().orElseThrow().inputs().getFirst();
        Tensor outputAdd = activationInput(outputGate);
        Tensor hiddenOutputGate = outputAdd.provenance().orElseThrow().inputs().get(1);
        return hiddenOutputGate.provenance().orElseThrow().inputs().getFirst();
    }

    private static LstmCell cell(DataType dataType, boolean bias) {
        Tensor inputWeight = tensor(dataType, Shape.of(16, 3), true);
        Tensor hiddenWeight = tensor(dataType, Shape.of(16, 4), true);
        return bias
                ? new LstmCell(inputWeight, hiddenWeight,
                        tensor(dataType, Shape.of(16), true))
                : new LstmCell(inputWeight, hiddenWeight);
    }

    private static List<String> names(LstmCell cell) {
        return cell.parameters().stream().map(Parameter::name).toList();
    }

    private static Tensor tensor(DataType dataType, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                dataType, shape, Optional.empty(), requiresGrad));
    }

    private static void assertContains(
            String first, String second, ThrowingConstruction construction) {
        String message = assertThrows(IllegalArgumentException.class, construction::run).getMessage();
        assertTrue(message.contains(first), message);
        assertTrue(message.contains(second), message);
    }

    private static AtomicLong nextTensorIdState() throws Exception {
        Field field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    @FunctionalInterface
    private interface ThrowingConstruction {
        void run();
    }

    private static final class Owner extends Module {
        private Owner(LstmCell cell) {
            child("cell", cell);
        }
    }
}
