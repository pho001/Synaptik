package io.github.pho001.synaptik.prepare;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.CompileConstantPlan;
import io.github.pho001.synaptik.compiler.CompileDiagnostics;
import io.github.pho001.synaptik.compiler.DerivativeGraphMetadata;
import io.github.pho001.synaptik.compiler.GradientPublicationBinding;
import io.github.pho001.synaptik.compiler.PublicationPlan;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.ForwardPublicationBinding;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlan;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CallerInput;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CreatedBuffer;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.InitializedBuffer;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.ExecutionStep;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.PublicationStep;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.RepresentationCreationStep;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class GraphPreparationTest {
    @Test
    void requiresEveryProducerAndExternalConsumerToDeclareCrossPartitionValues() {
        Fixture fixture = fixture();
        ValueId crossing = new ValueId(2);
        var missingProducer = List.<PartitionPreparation<?, ?>>of(
                selectivePreparation(0, fixture, java.util.Set.of(new ValueId(0))),
                selectivePreparation(1, fixture,
                        java.util.Set.of(new ValueId(1), crossing, new ValueId(3))));
        var missingConsumer = List.<PartitionPreparation<?, ?>>of(
                selectivePreparation(0, fixture, java.util.Set.of(new ValueId(0), crossing)),
                selectivePreparation(1, fixture,
                        java.util.Set.of(new ValueId(1), new ValueId(3))));

        assertAll(
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> GraphPreparation.prepare(fixture.artifacts, missingProducer,
                                GraphPreparationTest::validSchedule)).getMessage()
                        .contains("cross-partition value ValueId[value=2] must be declared")),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> GraphPreparation.prepare(fixture.artifacts, missingConsumer,
                                GraphPreparationTest::validSchedule)).getMessage()
                        .contains("cross-partition value ValueId[value=2] must be declared")));
    }

    @Test
    void preparesAllContextsBeforeOrderedAnalysisAndFinalizationThenAssemblesOnce() {
        Fixture fixture = fixture();
        List<String> events = new ArrayList<>();
        List<PrepareContext<FakeInputs>> contexts = new ArrayList<>();
        AtomicInteger assemblerCalls = new AtomicInteger();

        List<PartitionPreparation<?, ?>> preparations = List.of(
                preparation(0, fixture, contexts, events),
                preparation(1, fixture, contexts, events));
        PreparedExecution execution = GraphPreparation.prepare(
                fixture.artifacts,
                preparations,
                context -> {
                    assemblerCalls.incrementAndGet();
                    events.add("assemble");
                    return validSchedule(context);
                });

        assertAll(
                () -> assertEquals(
                        List.of("analyze-0", "analyze-1", "finalize-0", "finalize-1", "assemble"),
                        events),
                () -> assertEquals(1, assemblerCalls.get()),
                () -> assertEquals(2, contexts.size()),
                () -> assertSame(fixture.partitions.get(0), contexts.get(0).partition()),
                () -> assertSame(fixture.partitions.get(1), contexts.get(1).partition()),
                () -> assertSame(
                        contexts.get(0).partitionDag().nodes(), contexts.get(0).nodes()),
                () -> assertSame(
                        contexts.get(1).partitionDag().nodes(), contexts.get(1).nodes()),
                () -> assertEquals(
                        List.of(fixture.nodes.get(0)), contexts.get(0).nodes()),
                () -> assertEquals(
                        List.of(fixture.nodes.get(1)), contexts.get(1).nodes()),
                () -> assertEquals(
                        List.of(fixture.values.get(0), fixture.values.get(2)),
                        contexts.get(0).values()),
                () -> assertEquals(
                        List.of(fixture.values.get(1), fixture.values.get(2), fixture.values.get(3)),
                        contexts.get(1).values()),
                () -> assertTrue(contexts.get(0).constants().isEmpty()),
                () -> assertSame(
                        fixture.constantValue,
                        contexts.get(1).constants().get(fixture.values.get(1).id())),
                () -> assertSame(execution.memoryPlan(), execution.schedule().memoryPlan()),
                () -> assertSame(
                        execution.memoryPlan(),
                        ((RepresentationCreationStep) execution.schedule().steps().getFirst())
                                .representationPlan().memoryPlan()));
    }

    @Test
    void validatesEarlierPhasesBeforeInvokingLaterCollaborators() {
        Fixture fixture = fixture();
        AtomicInteger analyses = new AtomicInteger();
        AtomicInteger finalizers = new AtomicInteger();
        AtomicInteger assemblers = new AtomicInteger();
        var preparation = new PartitionPreparation<>(
                new FakeInputs("target"),
                context -> {
                    analyses.incrementAndGet();
                    return new BackendPartitionAnalysis<>(
                            context.partition(),
                            new FakePlan("route"),
                            declarations(context));
                },
                new BackendPartitionFinalizer<FakePlan>() {
                    @Override
                    public BackendId backendId() {
                        return fixture.partitions.getFirst().owner();
                    }

                    @Override
                    public PreparedExecutable finalizePartition(
                            BackendPartitionFinalization<FakePlan> finalization) {
                        finalizers.incrementAndGet();
                        return new TestExecutable(finalization.memoryPlan());
                    }
                });

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "artifacts",
                        () -> GraphPreparation.prepare(null, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "preparations",
                        () -> GraphPreparation.prepare(fixture.artifacts, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "scheduleAssembler",
                        () -> GraphPreparation.prepare(fixture.artifacts, List.of(), null)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "preparations size must equal compile partition count 2",
                        () -> GraphPreparation.prepare(
                                fixture.artifacts, List.of(preparation), context -> null)),
                () -> assertEquals(0, analyses.get()),
                () -> assertEquals(0, finalizers.get()),
                () -> assertEquals(0, assemblers.get()));

        var nullAnalysis = new PartitionPreparation<>(
                new FakeInputs("target"), context -> null, preparation.finalizer());
        assertFailure(
                NullPointerException.class,
                "preparations[0].preparer returned null",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        List.of(nullAnalysis, preparation),
                        context -> {
                            assemblers.incrementAndGet();
                            return null;
                        }));
        assertAll(
                () -> assertEquals(0, finalizers.get()),
                () -> assertEquals(0, assemblers.get()));
    }

    @Test
    void projectsEveryContextBeforeInvokingTheFirstPreparer() {
        Fixture fixture = fixture(
                descriptor(),
                new TensorDescriptor(
                        DataType.FLOAT32,
                        Shape.ofDimensions(new DynamicDimension("N")),
                        Optional.empty(),
                        false));
        AtomicInteger analyses = new AtomicInteger();
        List<PartitionPreparation<?, ?>> preparations = List.of(
                countingPreparation(0, fixture, analyses),
                countingPreparation(1, fixture, analyses));

        IllegalArgumentException failure = assertThrows(
                IllegalArgumentException.class,
                () -> GraphPreparation.prepare(
                        fixture.artifacts, preparations, GraphPreparationTest::validSchedule));

        assertAll(
                () -> assertEquals(
                        "values[0].descriptor.shape must be fully static: Shape[N]",
                        failure.getMessage()),
                () -> assertEquals(0, analyses.get()));
    }

    @Test
    void rejectsInvalidSchedulesWithExactMessages() {
        Fixture fixture = fixture();
        List<PartitionPreparation<?, ?>> preparations = List.of(
                preparation(0, fixture, new ArrayList<>(), new ArrayList<>()),
                preparation(1, fixture, new ArrayList<>(), new ArrayList<>()));

        assertFailure(
                NullPointerException.class,
                "scheduleAssembler returned null",
                () -> GraphPreparation.prepare(fixture.artifacts, preparations, context -> null));
        assertFailure(
                IllegalArgumentException.class,
                "schedule memory plan does not match prepared memory plan",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> new PreparedSchedule(
                                new PreparedMemoryPlan(List.of(), List.of()), List.of())));
        assertFailure(
                IllegalArgumentException.class,
                "non-empty prepared memory plan requires a representation creation occurrence",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> new PreparedSchedule(context.memoryPlan(), List.of(
                                new ExecutionStep(context.partitions().get(0).executable()),
                                new ExecutionStep(context.partitions().get(1).executable()),
                                new PublicationStep(new PreparedPublication(
                                        context.memoryPlan(), 3, 0, 0))))));
        assertFailure(
                IllegalArgumentException.class,
                "caller-input occurrence count must equal bindable input count 1",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> scheduleWithPreparations(
                                context,
                                List.of(
                                        List.of(new CreatedBuffer(FakeBuffer::new)),
                                        List.of(new CreatedBuffer(FakeBuffer::new)),
                                        List.of(new InitializedBuffer(FakeBuffer::new)),
                                        List.of(new CreatedBuffer(FakeBuffer::new))))));
        assertFailure(
                IllegalArgumentException.class,
                "constant source ValueId[value=1] requires an initialized buffer representation",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> scheduleWithPreparations(
                                context,
                                List.of(
                                        List.of(new CallerInput()),
                                        List.of(new CreatedBuffer(FakeBuffer::new)),
                                        List.of(new CreatedBuffer(FakeBuffer::new)),
                                        List.of(new CreatedBuffer(FakeBuffer::new))))));
        assertFailure(
                IllegalArgumentException.class,
                "constant source ValueId[value=1] must not use a caller-input representation",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> scheduleWithPreparations(
                                context,
                                List.of(
                                        List.of(new CallerInput()),
                                        List.of(new CreatedBuffer(FakeBuffer::new)),
                                        List.of(
                                                new InitializedBuffer(FakeBuffer::new),
                                                new CallerInput()),
                                        List.of(new CreatedBuffer(FakeBuffer::new))))));
        assertFailure(
                IllegalArgumentException.class,
                "buffer ValueId[value=2] has an initialized representation but is not a constant source",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> scheduleWithPreparations(
                                context,
                                List.of(
                                        List.of(new CallerInput()),
                                        List.of(new InitializedBuffer(FakeBuffer::new)),
                                        List.of(new InitializedBuffer(FakeBuffer::new)),
                                        List.of(new CreatedBuffer(FakeBuffer::new))))));
        assertFailure(
                IllegalArgumentException.class,
                "execution occurrence count must equal prepared partition count 2",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> new PreparedSchedule(
                                context.memoryPlan(),
                                List.of(
                                        new RepresentationCreationStep(validRepresentations(context)),
                                        new ExecutionStep(
                                                context.partitions().getFirst().executable()),
                                        publication(context, 3, 0)))));
        assertFailure(
                IllegalArgumentException.class,
                "steps[3] publication representationIndex out of creation-plan range: 1",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> new PreparedSchedule(
                                context.memoryPlan(),
                                List.of(
                                        new RepresentationCreationStep(validRepresentations(context)),
                                        new ExecutionStep(
                                                context.partitions().get(0).executable()),
                                        new ExecutionStep(
                                                context.partitions().get(1).executable()),
                                        publication(context, 3, 1)))));
        assertFailure(
                IllegalArgumentException.class,
                "publication occurrence count must equal requested result count 1",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> new PreparedSchedule(
                                context.memoryPlan(),
                                List.of(
                                        new RepresentationCreationStep(validRepresentations(context)),
                                        new ExecutionStep(
                                                context.partitions().get(0).executable()),
                                        new ExecutionStep(
                                                context.partitions().get(1).executable())))));
        assertFailure(
                IllegalArgumentException.class,
                "publication occurrence 0 bufferIndex does not match requested value ValueId[value=3]",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> new PreparedSchedule(
                                context.memoryPlan(),
                                List.of(
                                        new RepresentationCreationStep(validRepresentations(context)),
                                        new ExecutionStep(
                                                context.partitions().get(0).executable()),
                                        new ExecutionStep(
                                                context.partitions().get(1).executable()),
                                        publication(context, 0, 0)))));
    }

    @Test
    void freshPreparationCallsProduceIndependentRecipesAndNeverInvokeCreators() {
        Fixture fixture = fixture();
        AtomicInteger creators = new AtomicInteger();
        List<PartitionPreparation<?, ?>> preparations = List.of(
                preparation(0, fixture, new ArrayList<>(), new ArrayList<>()),
                preparation(1, fixture, new ArrayList<>(), new ArrayList<>()));
        PreparedScheduleAssembler assembler = context -> {
            PreparedRepresentationPlan representations = new PreparedRepresentationPlan(
                    context.memoryPlan(),
                    List.of(
                            List.of(new CallerInput()),
                            List.of(new CreatedBuffer(() -> {
                                creators.incrementAndGet();
                                return new FakeBuffer();
                            })),
                            List.of(new InitializedBuffer(() -> {
                                creators.incrementAndGet();
                                return new FakeBuffer();
                            })),
                            List.of(new CreatedBuffer(() -> {
                                creators.incrementAndGet();
                                return new FakeBuffer();
                            }))),
                    List.of());
            return completeSchedule(context, representations);
        };

        PreparedExecution first = GraphPreparation.prepare(fixture.artifacts, preparations, assembler);
        PreparedExecution second = GraphPreparation.prepare(fixture.artifacts, preparations, assembler);

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotSame(first.memoryPlan(), second.memoryPlan()),
                () -> assertNotSame(first.schedule(), second.schedule()),
                () -> assertEquals(0, creators.get()));
    }

    @Test
    void failsClosedWhenARequestedPassThroughValueHasNoPreparedAssignment() {
        CompileArtifacts artifacts = passThroughArtifacts();

        assertFailure(
                IllegalArgumentException.class,
                "requested value has no prepared buffer assignment: ValueId[value=50]",
                () -> GraphPreparation.prepare(
                        artifacts,
                        List.of(),
                        context -> new PreparedSchedule(context.memoryPlan(), List.of())));
    }

    @Test
    void preservesAliasedForwardAndRepeatedGradientPublicationOccurrences() {
        Fixture fixture = fixture();
        CompileArtifacts artifacts = withAliasedResults(fixture.artifacts());
        List<PartitionPreparation<?, ?>> preparations = List.of(
                preparation(0, fixture, new ArrayList<>(), new ArrayList<>()),
                preparation(1, fixture, new ArrayList<>(), new ArrayList<>()));

        PreparedExecution execution = GraphPreparation.prepare(
                artifacts,
                preparations,
                context -> {
                    PreparedRepresentationPlan representations = validRepresentations(context);
                    return new PreparedSchedule(
                            context.memoryPlan(),
                            List.of(
                                    new RepresentationCreationStep(representations),
                                    new ExecutionStep(
                                            context.partitions().get(0).executable()),
                                    new ExecutionStep(
                                            context.partitions().get(1).executable()),
                                    new PublicationStep(new PreparedPublication(
                                            context.memoryPlan(), 3, 0, 0)),
                                    new PublicationStep(new PreparedPublication(
                                            context.memoryPlan(), 3, 0, 1)),
                                    new PublicationStep(new PreparedPublication(
                                            context.memoryPlan(), 3, 0, 2))));
                });

        assertAll(
                () -> assertEquals(3, execution.schedule().publicationCount()),
                () -> assertEquals(
                        List.of(3, 3, 3),
                        execution.schedule().steps().stream()
                                .filter(PublicationStep.class::isInstance)
                                .map(PublicationStep.class::cast)
                                .map(step -> step.publication().bufferIndex())
                                .toList()));
    }

    private static PartitionPreparation<FakeInputs, FakePlan> preparation(
            int index,
            Fixture fixture,
            List<PrepareContext<FakeInputs>> contexts,
            List<String> events) {
        BackendId owner = fixture.partitions.get(index).owner();
        return new PartitionPreparation<>(
                new FakeInputs("target-" + index),
                context -> {
                    contexts.add(context);
                    events.add("analyze-" + index);
                    return new BackendPartitionAnalysis<>(
                            context.partition(),
                            new FakePlan("route-" + index),
                            declarations(context));
                },
                new BackendPartitionFinalizer<>() {
                    @Override
                    public BackendId backendId() {
                        return owner;
                    }

                    @Override
                    public PreparedExecutable finalizePartition(
                            BackendPartitionFinalization<FakePlan> finalization) {
                        events.add("finalize-" + index);
                        return new TestExecutable(finalization.memoryPlan());
                    }
                });
    }

    private static PartitionPreparation<FakeInputs, FakePlan> countingPreparation(
            int index, Fixture fixture, AtomicInteger analyses) {
        BackendId owner = fixture.partitions.get(index).owner();
        return new PartitionPreparation<>(
                new FakeInputs("target-" + index),
                context -> {
                    analyses.incrementAndGet();
                    return new BackendPartitionAnalysis<>(
                            context.partition(), new FakePlan("route"), declarations(context));
                },
                new BackendPartitionFinalizer<>() {
                    @Override
                    public BackendId backendId() {
                        return owner;
                    }

                    @Override
                    public PreparedExecutable finalizePartition(
                            BackendPartitionFinalization<FakePlan> finalization) {
                        return new TestExecutable(finalization.memoryPlan());
                    }
                });
    }

    private static PartitionPreparation<FakeInputs, FakePlan> selectivePreparation(
            int index, Fixture fixture, java.util.Set<ValueId> declaredValues) {
        BackendId owner = fixture.partitions.get(index).owner();
        return new PartitionPreparation<>(new FakeInputs("selective-" + index), context ->
                new BackendPartitionAnalysis<>(context.partition(), new FakePlan("route"),
                        context.values().stream().filter(value -> declaredValues.contains(value.id()))
                                .map(value -> (PreparationResourceRequirement)
                                        new PreparationResourceRequirement.Buffer(value.id(), 24, 8))
                                .toList()),
                new BackendPartitionFinalizer<>() {
                    @Override public BackendId backendId() { return owner; }
                    @Override public PreparedExecutable finalizePartition(
                            BackendPartitionFinalization<FakePlan> finalization) {
                        return new TestExecutable(finalization.memoryPlan());
                    }
                });
    }

    private static List<PreparationResourceRequirement> declarations(
            PrepareContext<FakeInputs> context) {
        return context.values().stream()
                .map(value -> (PreparationResourceRequirement)
                        new PreparationResourceRequirement.Buffer(value.id(), 24, 8))
                .toList();
    }

    private static PreparedSchedule validSchedule(PreparedScheduleContext context) {
        return completeSchedule(context, validRepresentations(context));
    }

    private static PreparedRepresentationPlan validRepresentations(
            PreparedScheduleContext context) {
        return new PreparedRepresentationPlan(
                context.memoryPlan(),
                List.of(
                        List.of(new CallerInput()),
                        List.of(new CreatedBuffer(FakeBuffer::new)),
                        List.of(new InitializedBuffer(FakeBuffer::new)),
                        List.of(new CreatedBuffer(FakeBuffer::new))),
                List.of());
    }

    private static PreparedSchedule scheduleWithPreparations(
            PreparedScheduleContext context,
            List<List<PreparedRepresentationPlan.BufferPreparation>> preparations) {
        return completeSchedule(
                context,
                new PreparedRepresentationPlan(context.memoryPlan(), preparations, List.of()));
    }

    private static PreparedSchedule completeSchedule(
            PreparedScheduleContext context, PreparedRepresentationPlan representations) {
        int outputBuffer = context.bufferAssignments().stream()
                .filter(assignment -> assignment.valueId().equals(new ValueId(3)))
                .findFirst()
                .orElseThrow()
                .planIndex();
        return new PreparedSchedule(
                context.memoryPlan(),
                List.of(
                        new RepresentationCreationStep(representations),
                        new ExecutionStep(context.partitions().get(0).executable()),
                        new ExecutionStep(context.partitions().get(1).executable()),
                        new PublicationStep(new PreparedPublication(
                                context.memoryPlan(), outputBuffer, 0, 0))));
    }

    private static PublicationStep publication(
            PreparedScheduleContext context, int bufferIndex, int representationIndex) {
        return new PublicationStep(new PreparedPublication(
                context.memoryPlan(), bufferIndex, representationIndex, 0));
    }

    static Fixture fixture() {
        return fixture(descriptor(), descriptor());
    }

    private static Fixture fixture(
            TensorDescriptor firstDescriptor, TensorDescriptor secondDescriptor) {
        ValueId input = new ValueId(0);
        ValueId constant = new ValueId(1);
        ValueId intermediate = new ValueId(2);
        ValueId output = new ValueId(3);
        List<GraphValue> values = List.of(
                new GraphValue(input, firstDescriptor),
                new GraphValue(constant, secondDescriptor),
                new GraphValue(intermediate, firstDescriptor),
                new GraphValue(output, secondDescriptor));
        CompiledNode first = node(10, List.of(input), intermediate);
        CompiledNode second = node(11, List.of(intermediate, constant), output);
        List<CompiledNode> nodes = List.of(first, second);
        CompiledGraphModel graph = new CompiledGraphModel(
                values,
                nodes,
                List.of(input, constant),
                List.of(output),
                Map.of(first.id(), GraphPhase.FORWARD, second.id(), GraphPhase.FORWARD));
        PlannedPartition firstPartition =
                new PlannedPartition(new BackendId("cpu"), List.of(first.id()));
        PlannedPartition secondPartition =
                new PlannedPartition(new BackendId("gpu"), List.of(second.id()));
        List<PlannedPartition> partitions = List.of(firstPartition, secondPartition);
        LogicalMemoryPlan memory = LogicalMemoryPlanning.plan(graph, partitions);
        ScalarValue scalar = ScalarValue.float32(2.0f);
        CompileConstantPlan constants = construct(
                CompileConstantPlan.class,
                new Class<?>[] {List.class, List.class},
                List.of(input),
                List.of(new CompileConstantPlan.ConstantSource(constant, scalar)));
        PublicationPlan publication = construct(
                PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class},
                graph,
                List.of(new ForwardPublicationBinding(new TensorId(100), output)),
                List.of());
        CompileDiagnostics diagnostics = construct(
                CompileDiagnostics.class,
                new Class<?>[] {List.class},
                List.of());
        var derivativeOrders = new LinkedHashMap<NodeId, Integer>();
        derivativeOrders.put(first.id(), 0);
        derivativeOrders.put(second.id(), 0);
        CompileArtifacts artifacts = new CompileArtifacts(
                CompileMode.FORWARD_ONLY,
                graph,
                partitions,
                memory,
                publication,
                constants,
                diagnostics,
                new DerivativeGraphMetadata(graph, derivativeOrders));
        return new Fixture(artifacts, partitions, nodes, values, scalar);
    }

    private static TensorDescriptor descriptor() {
        return new TensorDescriptor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
    }

    private static CompiledNode node(long id, List<ValueId> inputs, ValueId output) {
        return new CompiledNode(
                new NodeId(id),
                new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE),
                inputs,
                List.of(output));
    }

    private static <T> T construct(
            Class<T> type, Class<?>[] parameterTypes, Object... arguments) {
        try {
            Constructor<T> constructor = type.getDeclaredConstructor(parameterTypes);
            constructor.setAccessible(true);
            return constructor.newInstance(arguments);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }

    private static CompileArtifacts passThroughArtifacts() {
        ValueId valueId = new ValueId(50);
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(new GraphValue(valueId, descriptor())),
                List.of(),
                List.of(valueId),
                List.of(valueId),
                Map.of());
        List<PlannedPartition> partitions = List.of();
        CompileConstantPlan constants = construct(
                CompileConstantPlan.class,
                new Class<?>[] {List.class, List.class},
                List.of(valueId),
                List.of());
        PublicationPlan publication = construct(
                PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class},
                graph,
                List.of(new ForwardPublicationBinding(new TensorId(200), valueId)),
                List.of());
        CompileDiagnostics diagnostics = construct(
                CompileDiagnostics.class, new Class<?>[] {List.class}, List.of());
        return new CompileArtifacts(
                CompileMode.FORWARD_ONLY,
                graph,
                partitions,
                LogicalMemoryPlanning.plan(graph, partitions),
                publication,
                constants,
                diagnostics,
                new DerivativeGraphMetadata(graph, Map.of()));
    }

    private static CompileArtifacts withAliasedResults(CompileArtifacts source) {
        ValueId output = source.graph().outputs().getFirst();
        PublicationPlan publication = construct(
                PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class},
                source.graph(),
                List.of(new ForwardPublicationBinding(new TensorId(300), output)),
                List.of(
                        new GradientPublicationBinding(1, 0, new TensorId(301), output),
                        new GradientPublicationBinding(1, 1, new TensorId(302), output)));
        return new CompileArtifacts(
                CompileMode.FORWARD_AND_BACKWARD,
                source.graph(),
                source.partitions(),
                source.memory(),
                publication,
                source.constants(),
                source.diagnostics(),
                source.derivatives());
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> type, String message, org.junit.jupiter.api.function.Executable action) {
        assertEquals(message, assertThrows(type, action).getMessage());
    }

    record Fixture(
            CompileArtifacts artifacts,
            List<PlannedPartition> partitions,
            List<CompiledNode> nodes,
            List<GraphValue> values,
            ScalarValue constantValue) {}

    private record FakeInputs(String target) implements BackendAnalysisInputs {}

    private record FakePlan(String route) implements BackendPreparationPlan {}

    private static final class FakeBuffer implements BufferRepresentation {
        @Override
        public void close() {}
    }

    private static final class TestExecutable extends PreparedExecutable {
        private TestExecutable(PreparedMemoryPlan memoryPlan) {
            super(memoryPlan, List.of(), List.of());
        }

        @Override
        protected boolean acceptsBufferRepresentation(
                int selectionIndex, BufferRepresentation representation) {
            return true;
        }

        @Override
        protected boolean acceptsWorkspaceRepresentation(
                int selectionIndex, WorkspaceRepresentation representation) {
            return true;
        }

        @Override
        protected BoundInvocation bindCompatible(
                RunState runState,
                BufferRepresentation[] bufferRepresentations,
                WorkspaceRepresentation[] workspaceRepresentations) {
            throw new UnsupportedOperationException();
        }
    }

    private enum SampleKind implements OperationKind {
        SAMPLE;

        private static final List<OperationSignature> SIGNATURES = List.of(
                new OperationSignature(
                        NoOperationAttrs.class, 0, Integer.MAX_VALUE, 1, Integer.MAX_VALUE));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
