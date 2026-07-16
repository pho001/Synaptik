package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.io.IOException;
import java.lang.reflect.Modifier;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class ForwardGraphOptimizationTest {
    @Test
    void exposesOnlyThePackagePrivateStatelessOptimizationContract() throws Exception {
        var method = ForwardGraphOptimization.class.getDeclaredMethod(
                "optimize", ValidatedGraph.class, GraphOptimizationConfig.class);
        var constructor = ForwardGraphOptimization.class.getDeclaredConstructor();

        assertAll(
                () -> assertTrue(Modifier.isFinal(ForwardGraphOptimization.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(
                        ForwardGraphOptimization.class.getModifiers())),
                () -> assertEquals(0, ForwardGraphOptimization.class.getDeclaredFields().length),
                () -> assertTrue(Modifier.isPrivate(constructor.getModifiers())),
                () -> assertTrue(Modifier.isStatic(method.getModifiers())),
                () -> assertFalse(Modifier.isPublic(method.getModifiers())),
                () -> assertSame(ValidatedGraph.class, method.getReturnType()),
                () -> assertEquals(1, Arrays.stream(
                                ForwardGraphOptimization.class.getDeclaredMethods())
                        .filter(declared -> !declared.isSynthetic())
                        .filter(declared -> !Modifier.isPrivate(declared.getModifiers()))
                        .count()),
                () -> assertEquals(1, GraphOptimizationConfig.class.getRecordComponents().length),
                () -> assertSame(boolean.class,
                        GraphOptimizationConfig.class.getRecordComponents()[0].getType()));
    }

    @Test
    void rejectsTopLevelNullsInSpecifiedOrderAndWithSpecifiedMessages() {
        CompiledGraphModel graph = passThroughGraph();
        ValidatedGraph validated = CapturedGraphInference.inferAndValidate(graph);

        assertAll(
                () -> assertEquals("validatedGraph", assertThrows(NullPointerException.class,
                        () -> ForwardGraphOptimization.optimize(null, null)).getMessage()),
                () -> assertEquals("optimizationConfig", assertThrows(NullPointerException.class,
                        () -> ForwardGraphOptimization.optimize(validated, null)).getMessage()));
    }

    @Test
    void disabledStillCanonicalizesValidatesAndRegeneratesConstraintNodeIds() {
        DynamicDimension inputExtent = new DynamicDimension("N");
        DynamicDimension outputExtent = new DynamicDimension("M");
        TensorDescriptor input = descriptor(Shape.ofDimensions(inputExtent), false);
        TensorDescriptor output = descriptor(Shape.ofDimensions(outputExtent), false);
        ValueId inputId = new ValueId(90);
        ValueId outputId = new ValueId(12);
        NodeId nodeId = new NodeId(41);
        Operation reshape = new Operation(
                ShapeTransformKind.RESHAPE, new TargetShapeAttrs(output.shape()));
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(new GraphValue(outputId, output), new GraphValue(inputId, input)),
                List.of(new CompiledNode(nodeId, reshape, List.of(inputId), List.of(outputId))),
                List.of(inputId),
                List.of(outputId),
                Map.of(nodeId, GraphPhase.FORWARD));
        ValidatedGraph incoming = CapturedGraphInference.inferAndValidate(graph);

        ValidatedGraph result = ForwardGraphOptimization.optimize(
                incoming, GraphOptimizationConfig.disabled());

        assertAll(
                () -> assertNotSame(incoming, result),
                () -> assertNotSame(graph, result.graph()),
                () -> assertEquals(List.of(new ValueId(0)), result.graph().inputs()),
                () -> assertEquals(List.of(new ValueId(1)), result.graph().outputs()),
                () -> assertEquals(new NodeId(0), result.graph().nodes().getFirst().id()),
                () -> assertSame(reshape, result.graph().nodes().getFirst().operation()),
                () -> assertSame(input, result.graph().values().get(0).descriptor()),
                () -> assertSame(output, result.graph().values().get(1).descriptor()),
                () -> assertEquals(List.of(nodeId), incoming.constraints().stream()
                        .map(DeferredGraphConstraint::nodeId).toList()),
                () -> assertEquals(List.of(new NodeId(0)), result.constraints().stream()
                        .map(DeferredGraphConstraint::nodeId).toList()),
                () -> assertNotSame(incoming.constraints(), result.constraints()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> result.constraints().clear()));
    }

    @Test
    void disabledSkipsOptionalWorkWhileStandardRunsTheBoundedSequence() {
        TensorDescriptor descriptor = descriptor(Shape.of(2), true);
        Operation dead = operation(UnaryElementwiseKind.ABS);
        Operation live = operation(UnaryElementwiseKind.ABS);
        Operation outputOperation = operation(UnaryElementwiseKind.NEG);
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(
                        new GraphValue(new ValueId(0), descriptor),
                        new GraphValue(new ValueId(1), descriptor),
                        new GraphValue(new ValueId(2), descriptor),
                        new GraphValue(new ValueId(3), descriptor)),
                List.of(
                        new CompiledNode(new NodeId(0), dead,
                                List.of(new ValueId(0)), List.of(new ValueId(1))),
                        new CompiledNode(new NodeId(1), live,
                                List.of(new ValueId(0)), List.of(new ValueId(2))),
                        new CompiledNode(new NodeId(2), outputOperation,
                                List.of(new ValueId(2)), List.of(new ValueId(3)))),
                List.of(new ValueId(0)),
                List.of(new ValueId(3)),
                Map.of(
                        new NodeId(0), GraphPhase.FORWARD,
                        new NodeId(1), GraphPhase.FORWARD,
                        new NodeId(2), GraphPhase.FORWARD));
        ValidatedGraph validated = CapturedGraphInference.inferAndValidate(graph);

        ValidatedGraph disabled = ForwardGraphOptimization.optimize(
                validated, GraphOptimizationConfig.disabled());
        ValidatedGraph standard = ForwardGraphOptimization.optimize(
                validated, GraphOptimizationConfig.standard());

        assertAll(
                () -> assertEquals(3, disabled.graph().nodes().size()),
                () -> assertSame(dead, disabled.graph().nodes().get(0).operation()),
                () -> assertEquals(2, standard.graph().nodes().size()),
                () -> assertSame(live, standard.graph().nodes().get(0).operation()),
                () -> assertNotSame(dead, standard.graph().nodes().get(0).operation()),
                () -> assertSame(outputOperation,
                        standard.graph().nodes().get(1).operation()),
                () -> assertEquals(List.of(new ValueId(2)), standard.graph().outputs()));
    }

    @Test
    void disabledSkipsExactRewritingWhileStandardAppliesItBeforeCleanup() {
        TensorDescriptor descriptor = descriptor(Shape.of(2), false);
        Operation identity = new Operation(
                ScalarElementwiseKind.MUL,
                new ScalarValueAttrs(ScalarValue.float32(1.0f)));
        Operation output = operation(UnaryElementwiseKind.NEG);
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(
                        new GraphValue(new ValueId(0), descriptor),
                        new GraphValue(new ValueId(1), descriptor),
                        new GraphValue(new ValueId(2), descriptor)),
                List.of(
                        new CompiledNode(new NodeId(0), identity,
                                List.of(new ValueId(0)), List.of(new ValueId(1))),
                        new CompiledNode(new NodeId(1), output,
                                List.of(new ValueId(1)), List.of(new ValueId(2)))),
                List.of(new ValueId(0)),
                List.of(new ValueId(2)),
                Map.of(
                        new NodeId(0), GraphPhase.FORWARD,
                        new NodeId(1), GraphPhase.FORWARD));
        ValidatedGraph incoming = CapturedGraphInference.inferAndValidate(graph);

        ValidatedGraph disabled = ForwardGraphOptimization.optimize(
                incoming, GraphOptimizationConfig.disabled());
        ValidatedGraph standard = ForwardGraphOptimization.optimize(
                incoming, GraphOptimizationConfig.standard());

        assertAll(
                () -> assertEquals(2, disabled.graph().nodes().size()),
                () -> assertSame(identity, disabled.graph().nodes().get(0).operation()),
                () -> assertEquals(1, standard.graph().nodes().size()),
                () -> assertSame(output, standard.graph().nodes().getFirst().operation()),
                () -> assertEquals(List.of(new ValueId(0)),
                        standard.graph().nodes().getFirst().inputs()));
    }

    @Test
    void preservesConstantRolesWhenDisabledAndFoldsBeforeSidecarCleanupWhenEnabled() {
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.INT32, Shape.of(2), Optional.empty(), false);
        Operation add = operation(BinaryArithmeticKind.ADD);
        Operation output = new Operation(
                ScalarElementwiseKind.ADD,
                new ScalarValueAttrs(ScalarValue.int32(0)));
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(
                        new GraphValue(new ValueId(10), descriptor),
                        new GraphValue(new ValueId(20), descriptor),
                        new GraphValue(new ValueId(30), descriptor),
                        new GraphValue(new ValueId(40), descriptor)),
                List.of(
                        new CompiledNode(new NodeId(50), add,
                                List.of(new ValueId(10), new ValueId(20)),
                                List.of(new ValueId(30))),
                        new CompiledNode(new NodeId(60), output,
                                List.of(new ValueId(30)), List.of(new ValueId(40)))),
                List.of(new ValueId(10), new ValueId(20)),
                List.of(new ValueId(40)),
                Map.of(
                        new NodeId(50), GraphPhase.FORWARD,
                        new NodeId(60), GraphPhase.FORWARD));
        CompileTimeConstantGraph sidecar = new CompileTimeConstantGraph(
                graph,
                Map.of(
                        new ValueId(10), new CompileTimeConstantGraph.Splat(ScalarValue.int32(2)),
                        new ValueId(20), new CompileTimeConstantGraph.Splat(ScalarValue.int32(3))));
        ValidatedGraph incoming = CapturedGraphInference.inferAndValidate(sidecar);

        ValidatedGraph disabled = ForwardGraphOptimization.optimize(
                incoming, GraphOptimizationConfig.disabled());
        ValidatedGraph standard = ForwardGraphOptimization.optimize(
                incoming, GraphOptimizationConfig.standard());

        assertAll(
                () -> assertEquals(2, disabled.graph().nodes().size()),
                () -> assertEquals(2, disabled.constants().size()),
                () -> assertEquals(List.of(), disabled.bindableInputs()),
                () -> assertEquals(List.of(new ValueId(0), new ValueId(1)),
                        disabled.graph().inputs()),
                () -> assertEquals(1, standard.graph().nodes().size()),
                () -> assertSame(output, standard.graph().nodes().getFirst().operation()),
                () -> assertEquals(List.of(new ValueId(0)), standard.graph().inputs()),
                () -> assertEquals(List.of(), standard.bindableInputs()),
                () -> assertEquals(5, standard.constants().get(new ValueId(0))
                        .value().int32Value()),
                () -> assertEquals(List.of(new ValueId(0)),
                        standard.graph().nodes().getFirst().inputs()),
                () -> assertSame(standard.constantGraph(),
                        CapturedGraphInference.inferAndValidate(
                                standard.constantGraph()).constantGraph()));
    }

    @Test
    void changedRewriteCandidateIsValidatedAndRegeneratesRetainedConstraints() {
        DynamicDimension inputExtent = new DynamicDimension("N");
        DynamicDimension outputExtent = new DynamicDimension("M");
        TensorDescriptor input = descriptor(Shape.ofDimensions(inputExtent), false);
        TensorDescriptor output = descriptor(Shape.ofDimensions(outputExtent), false);
        Operation identity = new Operation(
                ScalarElementwiseKind.MUL,
                new ScalarValueAttrs(ScalarValue.float32(1.0f)));
        Operation reshape = new Operation(
                ShapeTransformKind.RESHAPE, new TargetShapeAttrs(output.shape()));
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(
                        new GraphValue(new ValueId(4), input),
                        new GraphValue(new ValueId(8), input),
                        new GraphValue(new ValueId(12), output)),
                List.of(
                        new CompiledNode(new NodeId(30), identity,
                                List.of(new ValueId(4)), List.of(new ValueId(8))),
                        new CompiledNode(new NodeId(40), reshape,
                                List.of(new ValueId(8)), List.of(new ValueId(12)))),
                List.of(new ValueId(4)),
                List.of(new ValueId(12)),
                Map.of(
                        new NodeId(30), GraphPhase.FORWARD,
                        new NodeId(40), GraphPhase.FORWARD));
        ValidatedGraph incoming = CapturedGraphInference.inferAndValidate(graph);

        ValidatedGraph result = ForwardGraphOptimization.optimize(
                incoming, GraphOptimizationConfig.standard());

        assertAll(
                () -> assertEquals(List.of(new NodeId(40)), incoming.constraints().stream()
                        .map(DeferredGraphConstraint::nodeId).toList()),
                () -> assertEquals(1, result.graph().nodes().size()),
                () -> assertSame(reshape, result.graph().nodes().getFirst().operation()),
                () -> assertEquals(List.of(new ValueId(0)),
                        result.graph().nodes().getFirst().inputs()),
                () -> assertEquals(List.of(new NodeId(0)), result.constraints().stream()
                        .map(DeferredGraphConstraint::nodeId).toList()));
    }

    @Test
    void dceRunsBeforeCseSoADeadEarlierDuplicateCannotBecomeRepresentative() {
        TensorDescriptor descriptor = descriptor(Shape.of(2), true);
        Operation deadEarlier = operation(UnaryElementwiseKind.ABS);
        Operation liveLater = operation(UnaryElementwiseKind.ABS);
        Operation output = operation(UnaryElementwiseKind.NEG);
        CompiledGraphModel graph = graphWithDuplicateBranch(
                descriptor, deadEarlier, liveLater, output);

        ValidatedGraph result = ForwardGraphOptimization.optimize(
                CapturedGraphInference.inferAndValidate(graph),
                GraphOptimizationConfig.standard());

        assertAll(
                () -> assertEquals(2, result.graph().nodes().size()),
                () -> assertSame(liveLater, result.graph().nodes().get(0).operation()),
                () -> assertNotSame(deadEarlier, result.graph().nodes().get(0).operation()),
                () -> assertSame(output, result.graph().nodes().get(1).operation()));
    }

    @Test
    void changedCseCandidateIsValidatedAndRegeneratesOnlyRetainedConstraints() {
        DynamicDimension inputExtent = new DynamicDimension("N");
        DynamicDimension outputExtent = new DynamicDimension("M");
        TensorDescriptor input = descriptor(Shape.ofDimensions(inputExtent), false);
        TensorDescriptor reshaped = descriptor(Shape.ofDimensions(outputExtent), false);
        Operation firstReshape = new Operation(
                ShapeTransformKind.RESHAPE, new TargetShapeAttrs(reshaped.shape()));
        Operation secondReshape = new Operation(
                ShapeTransformKind.RESHAPE, new TargetShapeAttrs(reshaped.shape()));
        Operation firstOutput = operation(UnaryElementwiseKind.NEG);
        Operation secondOutput = operation(UnaryElementwiseKind.NEG);
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(
                        new GraphValue(new ValueId(0), input),
                        new GraphValue(new ValueId(1), reshaped),
                        new GraphValue(new ValueId(2), reshaped),
                        new GraphValue(new ValueId(3), reshaped),
                        new GraphValue(new ValueId(4), reshaped)),
                List.of(
                        new CompiledNode(new NodeId(10), firstReshape,
                                List.of(new ValueId(0)), List.of(new ValueId(1))),
                        new CompiledNode(new NodeId(20), secondReshape,
                                List.of(new ValueId(0)), List.of(new ValueId(2))),
                        new CompiledNode(new NodeId(30), firstOutput,
                                List.of(new ValueId(1)), List.of(new ValueId(3))),
                        new CompiledNode(new NodeId(40), secondOutput,
                                List.of(new ValueId(2)), List.of(new ValueId(4)))),
                List.of(new ValueId(0)),
                List.of(new ValueId(3), new ValueId(4)),
                Map.of(
                        new NodeId(10), GraphPhase.FORWARD,
                        new NodeId(20), GraphPhase.FORWARD,
                        new NodeId(30), GraphPhase.FORWARD,
                        new NodeId(40), GraphPhase.FORWARD));
        ValidatedGraph incoming = CapturedGraphInference.inferAndValidate(graph);

        ValidatedGraph result = ForwardGraphOptimization.optimize(
                incoming, GraphOptimizationConfig.standard());

        assertAll(
                () -> assertEquals(List.of(new NodeId(10), new NodeId(20)),
                        incoming.constraints().stream()
                                .map(DeferredGraphConstraint::nodeId).toList()),
                () -> assertEquals(3, result.graph().nodes().size()),
                () -> assertEquals(List.of(new NodeId(0)), result.constraints().stream()
                        .map(DeferredGraphConstraint::nodeId).toList()),
                () -> assertEquals(List.of(new ValueId(1)),
                        result.graph().nodes().get(1).inputs()),
                () -> assertEquals(List.of(new ValueId(1)),
                        result.graph().nodes().get(2).inputs()));
    }

    @Test
    void isDeterministicAndDoesNotMutateItsInput() {
        TensorDescriptor descriptor = descriptor(Shape.of(2), true);
        CompiledGraphModel graph = graphWithDuplicateBranch(
                descriptor,
                operation(UnaryElementwiseKind.ABS),
                operation(UnaryElementwiseKind.ABS),
                operation(UnaryElementwiseKind.NEG));
        ValidatedGraph incoming = CapturedGraphInference.inferAndValidate(graph);

        ValidatedGraph first = ForwardGraphOptimization.optimize(
                incoming, GraphOptimizationConfig.standard());
        ValidatedGraph second = ForwardGraphOptimization.optimize(
                incoming, GraphOptimizationConfig.standard());

        assertAll(
                () -> assertEquals(first, second),
                () -> assertNotSame(first.graph(), second.graph()),
                () -> assertEquals(3, graph.nodes().size()),
                () -> assertEquals(List.of(new ValueId(3)), graph.outputs()),
                () -> assertThrows(UnsupportedOperationException.class,
                        () -> first.graph().nodes().clear()));
    }

    @Test
    void canonicalCandidateValidationFailurePropagatesWithTask0002Context() {
        TensorDescriptor input = descriptor(Shape.of(2), true);
        TensorDescriptor invalidOutput = descriptor(Shape.of(3), true);
        CompiledGraphModel invalid = new CompiledGraphModel(
                List.of(
                        new GraphValue(new ValueId(8), input),
                        new GraphValue(new ValueId(2), invalidOutput)),
                List.of(new CompiledNode(
                        new NodeId(7),
                        operation(UnaryElementwiseKind.ABS),
                        List.of(new ValueId(8)),
                        List.of(new ValueId(2)))),
                List.of(new ValueId(8)),
                List.of(new ValueId(2)),
                Map.of(new NodeId(7), GraphPhase.FORWARD));

        String message = assertThrows(IllegalArgumentException.class,
                () -> ForwardGraphOptimization.optimize(
                        new ValidatedGraph(invalid, List.of()),
                        GraphOptimizationConfig.disabled())).getMessage();

        assertAll(
                () -> assertTrue(message.startsWith("nodes[0] NodeId[value=0] ")),
                () -> assertTrue(message.contains("output[0] ValueId[value=1] expected=")),
                () -> assertTrue(message.contains("stored=TensorDescriptor")));
    }

    @Test
    void sourceLocksOneShotCanonicalizeValidateRewriteFoldDceCseDceOrder() throws IOException {
        String source = Files.readString(findOptimizationSource());
        int canonicalize = source.indexOf("GraphCanonicalization.canonicalize(");
        int rewrite = source.indexOf("ForwardExactArithmeticRewriting.rewrite(");
        int fold = source.indexOf("ForwardConstantFolding.fold(");
        int firstDce = source.indexOf("ForwardDeadCodeElimination.eliminate(");
        int cse = source.indexOf("ForwardCommonSubexpressionElimination.eliminate(");
        int secondDce = source.indexOf(
                "ForwardDeadCodeElimination.eliminate(", firstDce + 1);

        assertAll(
                () -> assertTrue(canonicalize >= 0),
                () -> assertTrue(rewrite > canonicalize),
                () -> assertTrue(fold > rewrite),
                () -> assertTrue(firstDce > fold),
                () -> assertTrue(cse > firstDce),
                () -> assertTrue(secondDce > cse),
                () -> assertEquals(-1, source.indexOf(
                        "ForwardDeadCodeElimination.eliminate(", secondDce + 1)),
                () -> assertEquals(-1, source.indexOf(
                        "ForwardCommonSubexpressionElimination.eliminate(", cse + 1)),
                () -> assertEquals(-1, source.indexOf(
                        "ForwardExactArithmeticRewriting.rewrite(", rewrite + 1)),
                () -> assertEquals(-1, source.indexOf(
                        "ForwardConstantFolding.fold(", fold + 1)),
                () -> assertFalse(source.contains("while (")),
                () -> assertFalse(source.contains("for (")),
                () -> assertTrue(source.contains(
                        "if (candidate == current.constantGraph())")),
                () -> assertEquals(2, occurrences(
                        source, "CapturedGraphInference.inferAndValidate(")));
    }

    private static CompiledGraphModel graphWithDuplicateBranch(
            TensorDescriptor descriptor,
            Operation deadEarlier,
            Operation liveLater,
            Operation output) {
        return new CompiledGraphModel(
                List.of(
                        new GraphValue(new ValueId(0), descriptor),
                        new GraphValue(new ValueId(1), descriptor),
                        new GraphValue(new ValueId(2), descriptor),
                        new GraphValue(new ValueId(3), descriptor)),
                List.of(
                        new CompiledNode(new NodeId(0), deadEarlier,
                                List.of(new ValueId(0)), List.of(new ValueId(1))),
                        new CompiledNode(new NodeId(1), liveLater,
                                List.of(new ValueId(0)), List.of(new ValueId(2))),
                        new CompiledNode(new NodeId(2), output,
                                List.of(new ValueId(2)), List.of(new ValueId(3)))),
                List.of(new ValueId(0)),
                List.of(new ValueId(3)),
                Map.of(
                        new NodeId(0), GraphPhase.FORWARD,
                        new NodeId(1), GraphPhase.FORWARD,
                        new NodeId(2), GraphPhase.FORWARD));
    }

    private static CompiledGraphModel passThroughGraph() {
        TensorDescriptor descriptor = descriptor(Shape.of(1), false);
        ValueId input = new ValueId(0);
        return new CompiledGraphModel(
                List.of(new GraphValue(input, descriptor)),
                List.of(),
                List.of(input),
                List.of(input),
                Map.of());
    }

    private static Path findOptimizationSource() {
        Path relative = Path.of(
                "modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/"
                        + "ForwardGraphOptimization.java");
        Path moduleRelative = Path.of(
                "src/main/java/io/github/pho001/synaptik/compiler/"
                        + "ForwardGraphOptimization.java");
        for (Path root = Path.of("").toAbsolutePath(); root != null; root = root.getParent()) {
            if (Files.isRegularFile(root.resolve(relative))) {
                return root.resolve(relative);
            }
            if (Files.isRegularFile(root.resolve(moduleRelative))) {
                return root.resolve(moduleRelative);
            }
        }
        throw new IllegalStateException("ForwardGraphOptimization.java source not found");
    }

    private static int occurrences(String source, String needle) {
        int count = 0;
        int offset = 0;
        while ((offset = source.indexOf(needle, offset)) >= 0) {
            count++;
            offset += needle.length();
        }
        return count;
    }

    private static Operation operation(OperationKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }

    private static TensorDescriptor descriptor(Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), requiresGrad);
    }
}
