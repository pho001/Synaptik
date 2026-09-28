package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_INT_UNALIGNED;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.loss.LossKind;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.operation.loss.MeanSquaredErrorAttrs;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalization;
import io.github.pho001.synaptik.prepare.BackendPartitionFinalizationResult;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.PartitionPreparation;
import io.github.pho001.synaptik.prepare.PreparationResourceAssignment;
import io.github.pho001.synaptik.prepare.PreparedBufferAssignment;
import io.github.pho001.synaptik.prepare.PreparedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.trace.TraceEvent;
import io.github.pho001.synaptik.trace.TraceLevel;
import io.github.pho001.synaptik.trace.TracePayload;
import io.github.pho001.synaptik.trace.TracePhase;
import io.github.pho001.synaptik.trace.payload.BackendInvocationOutcome;
import io.github.pho001.synaptik.trace.payload.BackendPreparationOutcome;
import io.github.pho001.synaptik.trace.payload.TraceNativeStatusKind;
import io.github.pho001.synaptik.trace.payload.TraceOutcomeStatus;
import io.github.pho001.synaptik.trace.payload.TraceRouteKind;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class MetalNegPreparedExecutionTest {
    private static final int MPSGRAPH_AUTORELEASE_STRESS_RUNS = 5_000;

    @Test
    void singletonRouteBoundaryIsChosenBeforeExactDeclarationsWithoutPhysicalAllocation() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        try {
            SingleNegRoute maximum = singleNegRoute(
                    context, Shape.of(0xffff_ffffL), Optional.empty());
            assertEquals(MetalPreparedRoute.CUSTOM_SINGLE_NEG,
                    maximum.analysis().plan().route());
            assertEquals(2, maximum.analysis().requirements().size());
            assertTrue(maximum.analysis().plan().addressWorkspace().isEmpty());

            SingleNegRoute firstOversized = singleNegRoute(
                    context, Shape.of(0x1_0000_0000L), Optional.empty());
            assertEquals(MetalPreparedRoute.MPSGRAPH,
                    firstOversized.analysis().plan().route());
            assertEquals(3, firstOversized.analysis().requirements().size());
            assertTrue(firstOversized.analysis().plan().addressWorkspace().isPresent());
            assertThrows(IllegalArgumentException.class,
                    () -> context.createMpsGraphExecutable(maximum.analysis().plan()));
            assertThrows(IllegalArgumentException.class,
                    () -> context.createNegKernelPipeline(firstOversized.analysis().plan()));

            SingleNegRoute splat = singleNegRoute(
                    context, Shape.of(4), Optional.of(ScalarValue.float32(-2.0f)));
            assertEquals(MetalPreparedRoute.CUSTOM_SINGLE_NEG,
                    splat.analysis().plan().route());
            assertTrue(splat.analysis().plan().feedSplats().getFirst().isPresent());
            assertEquals(0, api.bufferCreates.get());
            assertEquals(0, api.pipelineCreates.get());
            assertEquals(0, api.executableCreates.get());
        } finally {
            context.close();
        }
    }

    @Test
    void authenticatedMpsGraphSelectionForSingletonFixesWorkspaceBeforeFinalization() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        try {
            SingleNegRoute heuristic = singleNegRoute(
                    context, Shape.of(4), Optional.empty());
            PrepareContext<MetalNegAnalysisInputs> source = new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, heuristic.analysis().plan().partitionDag(), List.of(
                    new GraphValue(heuristic.feed(), descriptor(Shape.of(4))),
                    new GraphValue(heuristic.target(), descriptor(Shape.of(4)))), List.of(
                    requirement(heuristic.feed(), descriptor(Shape.of(4)),
                            Optional.empty(), List.of(heuristic.partition()), false),
                    requirement(heuristic.target(), descriptor(Shape.of(4)),
                            Optional.of(heuristic.partition()), List.of(), true)), Map.of(), new MetalNegAnalysisInputs(context));
            var generator = new MetalNegRouteCandidateGenerator();
            var batch = generator.generate(source, heuristic.analysis().plan(), 2);
            var handoff = generator.presentHandoff(
                    heuristic.partition(), batch, MetalNegTuningBatch.Candidate.MPSGRAPH);
            BackendPartitionAnalysis<MetalNegPreparationPlan> selected =
                    new MetalNegPartitionPreparer().analyze(new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, source.partitionDag(), source.values(), source.memoryRequirements(), source.constants(), new MetalNegAnalysisInputs(context, Optional.of(handoff))));
            assertEquals(MetalPreparedRoute.MPSGRAPH, selected.plan().route());
            assertEquals(3, selected.requirements().size());
            assertTrue(selected.plan().addressWorkspace().isPresent());

            FinalizationFixture assignment = finalization(selected);
            BackendPartitionFinalizationResult finalized =
                    new MetalNegPartitionFinalizer(context)
                            .finalizePartition(assignment.finalization());
            try {
                assertTrue(finalized.resources().getFirst()
                        instanceof MetalMpsGraphExecutableResource);
                assertEquals(1, api.executableCreates.get());
                assertEquals(0, api.pipelineCreates.get());
            } finally {
                finalized.resources().forEach(resource -> resource.close());
            }
        } finally {
            context.close();
        }
    }

    @Test
    void forcedSingletonRoutesIndependentlySatisfyTheStrictModelNegContractAndStayFixed() {
        RecordingNativeApi api = new RecordingNativeApi();
        api.emulateSingletonMpsGraphNeg = true;
        MetalDeviceContext context = MetalDeviceContext.open(api);
        List<TraceEvent<? extends TracePayload>> customEvents = new CopyOnWriteArrayList<>();
        List<TraceEvent<? extends TracePayload>> graphEvents = new CopyOnWriteArrayList<>();
        SingleNegRoute customRoute = singleNegRoute(
                context,
                Shape.of(6),
                Optional.empty(),
                new MetalTraceProducer(customEvents::add),
                MetalPreparedRoute.CUSTOM_SINGLE_NEG);
        SingleNegRoute graphRoute = singleNegRoute(
                context,
                Shape.of(6),
                Optional.empty(),
                new MetalTraceProducer(graphEvents::add),
                MetalPreparedRoute.MPSGRAPH);
        PreparedExecution customExecution = null;
        PreparedExecution graphExecution = null;
        MetalBufferRepresentation input = null;
        try {
            assertSame(MetalPreparedRoute.CUSTOM_SINGLE_NEG,
                    customRoute.analysis().plan().route());
            assertSame(MetalPreparedRoute.MPSGRAPH, graphRoute.analysis().plan().route());
            assertTrue(customRoute.analysis().plan().addressWorkspace().isEmpty());
            assertTrue(graphRoute.analysis().plan().addressWorkspace().isPresent());

            customExecution = prepareSingleExecution(context, customRoute);
            graphExecution = prepareSingleExecution(context, graphRoute);
            assertEquals(1, api.pipelineCreates.get());
            assertEquals(1, api.executableCreates.get());
            assertEquals(TraceRouteKind.CUSTOM_KERNEL,
                    ((BackendPreparationOutcome) customEvents.getLast().payload()).route());
            assertEquals(TraceRouteKind.GRAPH_EXECUTABLE,
                    ((BackendPreparationOutcome) graphEvents.getLast().payload()).route());

            input = context.createBuffer(6L * Float.BYTES);
            uploadBits(input, 0x3f800000, 0xc0000000, 0x00000000,
                    0x80000000, 0x7f800000, 0x7fc00000);
            var runner = new PreparedExecutionRunner();
            var customResult = runner.run(customExecution, List.of(input));
            try (Arena arena = Arena.ofConfined()) {
                assertNegated(
                        (MetalBufferRepresentation) customResult.publicationRepresentation(0),
                        new float[] {-1.0f, 2.0f, -0.0f, 0.0f,
                                Float.NEGATIVE_INFINITY, Float.NaN},
                        arena);
            } finally {
                customResult.close();
            }
            var graphResult = runner.run(graphExecution, List.of(input));
            try (Arena arena = Arena.ofConfined()) {
                assertNegated(
                        (MetalBufferRepresentation) graphResult.publicationRepresentation(0),
                        new float[] {-1.0f, 2.0f, -0.0f, 0.0f,
                                Float.NEGATIVE_INFINITY, Float.NaN},
                        arena);
            } finally {
                graphResult.close();
            }

            MetalInvocationPlan customInvocation = customEvents.stream()
                    .map(TraceEvent::payload)
                    .filter(MetalInvocationPlan.class::isInstance)
                    .map(MetalInvocationPlan.class::cast)
                    .findFirst()
                    .orElseThrow();
            MetalInvocationPlan graphInvocation = graphEvents.stream()
                    .map(TraceEvent::payload)
                    .filter(MetalInvocationPlan.class::isInstance)
                    .map(MetalInvocationPlan.class::cast)
                    .findFirst()
                    .orElseThrow();
            long tensorBytes = 6L * Float.BYTES;
            assertEquals(tensorBytes, customInvocation.aggregateInputBytes());
            assertEquals(tensorBytes, customInvocation.aggregateOutputBytes());
            assertEquals(0L, customInvocation.aggregateInternalBytes());
            assertEquals(0L, customInvocation.aggregateSplatBytes());
            assertEquals(0L, customInvocation.aggregateWorkspaceBytes());
            assertEquals(2L * tensorBytes, customInvocation.aggregateRequiredBytes());
            assertEquals(0, customInvocation.workspaceCount());
            long workspaceBytes = 2L * ADDRESS.byteSize();
            assertEquals(tensorBytes, graphInvocation.aggregateInputBytes());
            assertEquals(tensorBytes, graphInvocation.aggregateOutputBytes());
            assertEquals(0L, graphInvocation.aggregateInternalBytes());
            assertEquals(0L, graphInvocation.aggregateSplatBytes());
            assertEquals(workspaceBytes, graphInvocation.aggregateWorkspaceBytes());
            assertEquals(
                    2L * tensorBytes + workspaceBytes,
                    graphInvocation.aggregateRequiredBytes());
            assertEquals(1, graphInvocation.workspaceCount());

            assertSame(MetalPreparedRoute.CUSTOM_SINGLE_NEG,
                    customRoute.analysis().plan().route());
            assertSame(MetalPreparedRoute.MPSGRAPH, graphRoute.analysis().plan().route());
            assertEquals(1, api.customRunCalls.get());
            assertEquals(1, api.runCalls.get());
        } finally {
            close(input);
            close(graphExecution);
            close(customExecution);
            context.close();
        }
    }

    @Test
    void forcingRejectsEveryUnapprovedCurrentRouteBeforeNativeConstruction() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        try {
            var preparer = new MetalNegPartitionPreparer();
            assertThrows(IllegalArgumentException.class,
                    () -> preparer.analyzeForTesting(
                            task0052MinContext(context), MetalPreparedRoute.MPSGRAPH));
            assertThrows(IllegalArgumentException.class,
                    () -> preparer.analyzeForTesting(
                            task0052MinContext(context),
                            MetalPreparedRoute.CUSTOM_SINGLE_NEG));
            assertThrows(IllegalArgumentException.class,
                    () -> singleNegRoute(
                            context,
                            Shape.of(6),
                            Optional.empty(),
                            null,
                            MetalPreparedRoute.CUSTOM_PROGRAM));
            assertEquals(0, api.pipelineCreates.get());
            assertEquals(0, api.executableCreates.get());
            assertEquals(0, api.customRunCalls.get());
            assertEquals(0, api.runCalls.get());
        } finally {
            context.close();
        }
    }

    @Test
    void forcedRouteCreateAndRunFailuresNeverAttemptTheOtherFamily() {
        RecordingNativeApi customApi = new RecordingNativeApi();
        customApi.pipelineCreateStatus = 12;
        MetalDeviceContext customContext = MetalDeviceContext.open(customApi);
        try {
            SingleNegRoute customRoute = singleNegRoute(
                    customContext,
                    Shape.of(2),
                    Optional.empty(),
                    null,
                    MetalPreparedRoute.CUSTOM_SINGLE_NEG);
            FinalizationFixture customAssignment = finalization(customRoute.analysis());
            assertThrows(
                    MetalNativeApi.NativeFailure.class,
                    () -> new MetalNegPartitionFinalizer(customContext)
                            .finalizePartition(customAssignment.finalization()));
            assertEquals(1, customApi.pipelineCreates.get());
            assertEquals(0, customApi.executableCreates.get());
            assertEquals(0, customApi.runCalls.get());
        } finally {
            customContext.close();
        }

        RecordingNativeApi graphApi = new RecordingNativeApi();
        graphApi.runStatus = 17;
        MetalDeviceContext graphContext = MetalDeviceContext.open(graphApi);
        PreparedExecution graphExecution = null;
        MetalBufferRepresentation input = null;
        try {
            SingleNegRoute graphRoute = singleNegRoute(
                    graphContext,
                    Shape.of(2),
                    Optional.empty(),
                    null,
                    MetalPreparedRoute.MPSGRAPH);
            graphExecution = prepareSingleExecution(graphContext, graphRoute);
            input = graphContext.createBuffer(2L * Float.BYTES);
            uploadBits(input, 0x3f800000, 0xc0000000);
            PreparedExecution prepared = graphExecution;
            MetalBufferRepresentation caller = input;
            assertThrows(
                    MetalNativeApi.NativeFailure.class,
                    () -> new PreparedExecutionRunner().run(prepared, List.of(caller)));
            assertSame(MetalPreparedRoute.MPSGRAPH, graphRoute.analysis().plan().route());
            assertEquals(1, graphApi.executableCreates.get());
            assertEquals(0, graphApi.pipelineCreates.get());
            assertEquals(1, graphApi.runCalls.get());
            assertEquals(0, graphApi.customRunCalls.get());
        } finally {
            close(input);
            close(graphExecution);
            graphContext.close();
        }
    }

    @Test
    void customCallerRouteUsesOnePipelineDirectOutputsAndOneDowncallPerRun() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        SingleNegRoute route = singleNegRoute(context, Shape.of(6), Optional.empty());
        FinalizationFixture assignment = finalization(route.analysis());
        BackendPartitionFinalizationResult finalized =
                new MetalNegPartitionFinalizer(context)
                        .finalizePartition(assignment.finalization());
        assertEquals(1, finalized.resources().size());
        assertTrue(finalized.resources().getFirst()
                instanceof MetalNegKernelPipelineResource);
        var schedule = new MetalNegPreparedScheduleAssembler(
                context, route.analysis().plan(), List.of(route.target()))
                .assembleRoute(assignment.memoryPlan(),
                        new PreparedPartition(route.partition(), finalized.executable()),
                        assignment.preparedAssignments());
        PreparedExecution execution = new PreparedExecution(
                assignment.memoryPlan(), schedule, finalized.resources());
        MetalBufferRepresentation input = context.createBuffer(24);
        try {
            uploadBits(input, 0x3f800000, 0xc0000000, 0x00000000,
                    0x80000000, 0x7f800000, 0x7fc12345);
            var runner = new PreparedExecutionRunner();
            var first = runner.run(execution, List.of(input));
            Object firstOutput = first.publicationRepresentation(0);
            try (Arena arena = Arena.ofConfined()) {
                assertNegated((MetalBufferRepresentation) firstOutput,
                        new float[] {-1.0f, 2.0f, -0.0f, 0.0f,
                                Float.NEGATIVE_INFINITY, Float.NaN}, arena);
            } finally {
                first.close();
            }
            var second = runner.run(execution, List.of(input));
            try {
                assertNotSame(firstOutput, second.publicationRepresentation(0));
            } finally {
                second.close();
            }
            assertEquals(1, api.pipelineCreates.get());
            assertEquals(0, api.executableCreates.get());
            assertEquals(2, api.customRunCalls.get());
            assertEquals(0, api.runCalls.get());
            assertTrue(assignment.memoryPlan().workspaces().isEmpty());
        } finally {
            execution.close();
            input.close();
            context.close();
        }
        assertEquals(1, api.pipelineReleases.get());
        assertEquals(1, api.contextReleases.get());
    }

    @Test
    void MetalTraceObserverRuntimeFailureDoesNotChangeOrRepeatBackendWork() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        AtomicInteger callbacks = new AtomicInteger();
        MetalTraceProducer producer = new MetalTraceProducer(event -> {
            if (callbacks.incrementAndGet() == 2) {
                throw new IllegalStateException("observer failure");
            }
        });
        SingleNegRoute route =
                singleNegRoute(context, Shape.of(2), Optional.empty(), producer);
        FinalizationFixture assignment = finalization(route.analysis());
        BackendPartitionFinalizationResult finalized =
                new MetalNegPartitionFinalizer(context)
                        .finalizePartition(assignment.finalization());
        var schedule = new MetalNegPreparedScheduleAssembler(
                context, route.analysis().plan(), List.of(route.target()))
                .assembleRoute(
                        assignment.memoryPlan(),
                        new PreparedPartition(route.partition(), finalized.executable()),
                        assignment.preparedAssignments());
        PreparedExecution execution = new PreparedExecution(
                assignment.memoryPlan(), schedule, finalized.resources());
        MetalBufferRepresentation input = context.createBuffer(2L * Float.BYTES);
        try {
            uploadBits(input, 0x3f800000, 0xc0000000);
            var runner = new PreparedExecutionRunner();
            try (var first = runner.run(execution, List.of(input));
                    Arena arena = Arena.ofConfined()) {
                assertNegated(
                        (MetalBufferRepresentation) first.publicationRepresentation(0),
                        new float[] {-1.0f, 2.0f},
                        arena);
            }
            try (var ignored = runner.run(execution, List.of(input))) {
                assertEquals(1, ignored.resultCount());
            }
            assertFalse(producer.enabled());
            assertEquals(2, callbacks.get());
            assertEquals(2, api.customRunCalls.get());
        } finally {
            execution.close();
            input.close();
            context.close();
        }
    }

    @Test
    void MetalTracePreparationObserverErrorPreservesMalformedNativeRollbackFailure() {
        RecordingNativeApi api = new RecordingNativeApi();
        api.pipelineCreateStatus = 12;
        api.createHandleOnFailure = true;
        api.pipelineReleaseStatus = 7;
        MetalDeviceContext context = MetalDeviceContext.open(api);
        List<TraceEvent<? extends TracePayload>> events = new CopyOnWriteArrayList<>();
        AssertionError observerFailure = new AssertionError("prepare observer");
        MetalTraceProducer producer = new MetalTraceProducer(event -> {
            events.add(event);
            if (event.payload() instanceof BackendPreparationOutcome) {
                throw observerFailure;
            }
        });
        SingleNegRoute route =
                singleNegRoute(context, Shape.of(2), Optional.empty(), producer);
        FinalizationFixture assignment = finalization(route.analysis());
        try {
            IllegalStateException failure = assertThrows(
                    IllegalStateException.class,
                    () -> new MetalNegPartitionFinalizer(context)
                            .finalizePartition(assignment.finalization()));
            assertEquals(
                    MetalNativeApi.NEG_KERNEL_PIPELINE_CREATE_OPERATION
                            + " returned failure with a non-null handle",
                    failure.getMessage());
            MetalNativeApi.NativeFailure nativeFailure =
                    (MetalNativeApi.NativeFailure) failure.getCause();
            assertNativeFailure(
                    nativeFailure, MetalNativeApi.NEG_KERNEL_PIPELINE_CREATE_OPERATION, 12);
            assertEquals(
                    MetalNativeApi.NEG_KERNEL_PIPELINE_CREATE_OPERATION
                            + " failed with native status KERNEL_COMPILATION_FAILED (12)",
                    nativeFailure.getMessage());
            assertEquals(2, failure.getSuppressed().length);
            MetalNativeApi.NativeFailure rollbackFailure =
                    (MetalNativeApi.NativeFailure) failure.getSuppressed()[0];
            assertNativeFailure(
                    rollbackFailure,
                    MetalNativeApi.NEG_KERNEL_PIPELINE_CREATE_OPERATION
                            + " malformed-handle cleanup",
                    7);
            assertEquals(
                    MetalNativeApi.NEG_KERNEL_PIPELINE_CREATE_OPERATION
                            + " malformed-handle cleanup failed with native status"
                            + " INTERNAL_ERROR (7)",
                    rollbackFailure.getMessage());
            assertSame(observerFailure, failure.getSuppressed()[1]);
            assertEquals(2, events.size());
            TraceEvent<? extends TracePayload> event = events.getLast();
            BackendPreparationOutcome outcome =
                    (BackendPreparationOutcome) event.payload();
            assertEquals(TracePhase.PREPARE, event.phase());
            assertEquals(TraceLevel.ERROR, event.level());
            assertEquals(TraceOutcomeStatus.FAILED, outcome.status());
            assertTrue(outcome.nativeStatus().isEmpty());
            assertEquals(1, api.pipelineCreates.get());
            assertEquals(1, api.pipelineReleases.get());
        } finally {
            context.close();
        }
    }

    @Test
    void MetalTraceSplatUploadFailureEmitsOnlyFailedPrepareAndRollsBack() {
        RecordingNativeApi api = new RecordingNativeApi();
        RuntimeException uploadFailure = new RuntimeException("prepared splat upload");
        api.failUploadCall = 1;
        api.uploadFailure = uploadFailure;
        MetalDeviceContext context = MetalDeviceContext.open(api);
        List<TraceEvent<? extends TracePayload>> events = new CopyOnWriteArrayList<>();
        MetalTraceProducer producer = new MetalTraceProducer(events::add);
        SingleNegRoute route = singleNegRoute(
                context,
                Shape.of(2),
                Optional.of(ScalarValue.float32(1.25f)),
                producer);
        FinalizationFixture assignment = finalization(route.analysis());
        try {
            RuntimeException actual = assertThrows(
                    RuntimeException.class,
                    () -> new MetalNegPartitionFinalizer(context)
                            .finalizePartition(assignment.finalization()));
            assertSame(uploadFailure, actual);
            assertEquals(2, events.size());
            TraceEvent<? extends TracePayload> event = events.getLast();
            BackendPreparationOutcome outcome =
                    (BackendPreparationOutcome) event.payload();
            assertEquals(TracePhase.PREPARE, event.phase());
            assertEquals(TraceLevel.ERROR, event.level());
            assertEquals(TraceOutcomeStatus.FAILED, outcome.status());
            assertEquals(0, api.customRunCalls.get());
            assertEquals(1, api.bufferCreates.get());
            assertEquals(1, api.bufferReleases.get());
            assertEquals(1, api.pipelineCreates.get());
            assertEquals(1, api.pipelineReleases.get());
            assertTrue(api.liveBufferHandles().isEmpty());
        } finally {
            context.close();
        }
    }

    @Test
    void MetalTraceCustomRunObserverErrorIsSuppressedOnNativeFailure() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        AssertionError observerFailure = new AssertionError("custom run observer");
        MetalTraceProducer producer = new MetalTraceProducer(event -> {
            if (event.payload() instanceof BackendInvocationOutcome) {
                throw observerFailure;
            }
        });
        SingleNegRoute route =
                singleNegRoute(context, Shape.of(2), Optional.empty(), producer);
        FinalizationFixture finalization = finalization(route.analysis());
        BackendPartitionFinalizationResult finalized =
                new MetalNegPartitionFinalizer(context)
                        .finalizePartition(finalization.finalization());
        var executable = (MetalNegPreparedExecutable) finalized.executable();
        var buffers = new ArrayList<MetalBufferRepresentation>();
        RunState state = null;
        try {
            for (var entry : finalization.memoryPlan().buffers()) {
                buffers.add(context.createBuffer(entry.byteSize()));
            }
            var bindings = buffers.stream()
                    .map(buffer -> List.of(new BufferRepresentationBinding(
                            buffer, RunResourceOwnership.BORROWED)))
                    .toList();
            state = new RunState(finalization.memoryPlan(), bindings, List.of());
            var invocation = executable.bind(state);
            api.customRunStatus = 11;

            MetalNativeApi.NativeFailure failure = assertThrows(
                    MetalNativeApi.NativeFailure.class, invocation::execute);

            assertNativeFailure(
                    failure, MetalNativeApi.NEG_KERNEL_PIPELINE_RUN_OPERATION, 11);
            assertEquals(
                    MetalNativeApi.NEG_KERNEL_PIPELINE_RUN_OPERATION
                            + " failed with native status EXECUTION_FAILED (11)",
                    failure.getMessage());
            assertArrayEquals(new Throwable[] {observerFailure}, failure.getSuppressed());
            assertEquals(1, api.customRunCalls.get());
            assertTrue(producer.enabled());
        } finally {
            if (state != null) state.close();
            buffers.forEach(MetalBufferRepresentation::close);
            finalized.resources().forEach(resource -> resource.close());
            context.close();
        }
    }

    @Test
    void customSplatRoutePersistsImmutableInputAndDefersReleaseUntilLastResult() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        SingleNegRoute route = singleNegRoute(
                context, Shape.of(2), Optional.of(ScalarValue.float32(-0.0f)));
        FinalizationFixture assignment = finalization(route.analysis());
        BackendPartitionFinalizationResult finalized =
                new MetalNegPartitionFinalizer(context)
                        .finalizePartition(assignment.finalization());
        assertEquals(2, finalized.resources().size());
        assertTrue(finalized.resources().get(0) instanceof MetalNegKernelPipelineResource);
        assertTrue(finalized.resources().get(1) instanceof MetalPreparedSplatResource);
        var schedule = new MetalNegPreparedScheduleAssembler(
                context, route.analysis().plan(), List.of(route.target()))
                .assembleRoute(assignment.memoryPlan(),
                        new PreparedPartition(route.partition(), finalized.executable()),
                        assignment.preparedAssignments());
        PreparedExecution execution = new PreparedExecution(
                assignment.memoryPlan(), schedule, finalized.resources());
        io.github.pho001.synaptik.runtime.run.RunResult first = null;
        io.github.pho001.synaptik.runtime.run.RunResult second = null;
        try {
            var runner = new PreparedExecutionRunner();
            first = runner.run(execution, List.of());
            second = runner.run(execution, List.of());
            MetalBufferRepresentation firstOutput =
                    (MetalBufferRepresentation) first.publicationRepresentation(0);
            MetalBufferRepresentation secondOutput =
                    (MetalBufferRepresentation) second.publicationRepresentation(0);
            long firstOutputHandle = firstOutput.executionHandle().carrier().address();
            long secondOutputHandle = secondOutput.executionHandle().carrier().address();
            try (Arena arena = Arena.ofConfined()) {
                assertNegated(firstOutput, new float[] {0.0f, 0.0f}, arena);
                assertNegated(secondOutput, new float[] {0.0f, 0.0f}, arena);
                assertNotSame(firstOutput, secondOutput);
            }
            assertEquals(1, api.pipelineCreates.get());
            assertEquals(2, api.customRunCalls.get());
            assertEquals(3, api.bufferCreates.get(),
                    "prepare owns one splat and each run owns one output");
            assertEquals(1, api.uploads.get());
            assertEquals(2, api.customInputHandles.size());
            assertEquals(api.customInputHandles.get(0), api.customInputHandles.get(1));
            long splatHandle = api.customInputHandles.getFirst();

            execution.close();
            assertEquals(1, api.pipelineReleases.get());
            assertEquals(Set.of(splatHandle, firstOutputHandle, secondOutputHandle),
                    api.liveBufferHandles());
            int createsAtClose = api.bufferCreates.get();
            assertThrows(IllegalStateException.class,
                    () -> runner.run(execution, List.of()));
            assertEquals(createsAtClose, api.bufferCreates.get());

            second.close();
            second = null;
            assertEquals(Set.of(splatHandle, firstOutputHandle), api.liveBufferHandles(),
                    "one open result retains the immutable prepared splat");
            first.close();
            first = null;
            assertTrue(api.liveBufferHandles().isEmpty(),
                    "the last result releases its output and the deferred prepared splat");
            assertEquals(3, api.bufferReleases.get());
        } finally {
            close(second);
            close(first);
            execution.close();
            context.close();
        }
    }

    @Test
    void outputAllocationFailureReleasesRunBindingAndLaterRunReusesPreparedSplat() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        PreparedExecution execution = prepareSingleExecution(
                context, Shape.of(2), Optional.of(ScalarValue.float32(2.0f)));
        try {
            assertEquals(1, api.bufferCreates.get());
            assertEquals(1, api.uploads.get());
            RuntimeException allocationFailure = new RuntimeException("output allocation");
            api.failBufferCreateCall = 2;
            api.bufferCreateFailure = allocationFailure;
            assertSame(allocationFailure, assertThrows(
                    RuntimeException.class,
                    () -> new PreparedExecutionRunner().run(execution, List.of())));
            assertEquals(0, api.customRunCalls.get());
            assertEquals(1, api.liveBufferHandles().size(),
                    "run rollback releases its child binding but not the prepared owner");

            api.failBufferCreateCall = -1;
            try (var result = new PreparedExecutionRunner().run(execution, List.of());
                    Arena arena = Arena.ofConfined()) {
                assertNegated(
                        (MetalBufferRepresentation) result.publicationRepresentation(0),
                        new float[] {-2.0f, -2.0f},
                        arena);
            }
            assertEquals(1, api.customRunCalls.get());
            assertEquals(3, api.bufferCreates.get(),
                    "failed and successful output allocation attempts retain one prepared splat");
            assertEquals(1, api.uploads.get());
        } finally {
            execution.close();
            context.close();
        }
        assertTrue(api.liveBufferHandles().isEmpty());
    }

    @Test
    void preparedSplatBindingIsReadableForTransferAndMaterializationButNeverWritable() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        MetalBackendRuntime runtime = new MetalBackendRuntime(context);
        SingleNegRoute route = singleNegRoute(
                context, Shape.of(2), Optional.of(ScalarValue.float32(-2.5f)));
        FinalizationFixture assignment = finalization(route.analysis());
        BackendPartitionFinalizationResult finalized =
                new MetalNegPartitionFinalizer(context)
                        .finalizePartition(assignment.finalization());
        MetalNegPreparedExecutable executable =
                (MetalNegPreparedExecutable) finalized.executable();
        var binding = executable.splatResource(route.feed()).orElseThrow().newRunBinding();
        TensorDescriptor descriptor = descriptor(Shape.of(2));
        try {
            assertTrue(runtime.acceptsStorageLayoutTransfer(binding, descriptor));
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment destination = arena.allocate(2L * Float.BYTES, Float.BYTES);
                runtime.bindStorageLayoutDownload(binding, descriptor).accept(destination);
                assertEquals(Float.floatToRawIntBits(-2.5f),
                        destination.getAtIndex(JAVA_INT, 0));
                assertEquals(Float.floatToRawIntBits(-2.5f),
                        destination.getAtIndex(JAVA_INT, 1));
            }
            byte[] canonical = runtime.copyToCanonicalHostBytes(binding, descriptor, 8L);
            assertEquals(Float.floatToRawIntBits(-2.5f),
                    java.nio.ByteBuffer.wrap(canonical).getInt(0));
            assertThrows(IllegalArgumentException.class,
                    () -> runtime.bindStorageLayoutUpload(binding, descriptor));

            for (int index = finalized.resources().size() - 1; index >= 0; index--) {
                finalized.resources().get(index).close();
            }
            assertEquals(0, api.bufferReleases.get(),
                    "the live read binding defers prepared-owner physical release");
            byte[] afterOwnerClose =
                    runtime.copyToCanonicalHostBytes(binding, descriptor, 8L);
            assertArrayEquals(canonical, afterOwnerClose);
            RuntimeException deferredRelease =
                    new RuntimeException("deferred prepared splat release");
            api.bufferReleaseFailures.add(deferredRelease);
            assertSame(deferredRelease,
                    assertThrows(RuntimeException.class, binding::close));
            assertFalse(runtime.acceptsStorageLayoutTransfer(binding, descriptor));
        } finally {
            binding.close();
            for (int index = finalized.resources().size() - 1; index >= 0; index--) {
                finalized.resources().get(index).close();
            }
            runtime.close();
        }
        assertEquals(1, api.bufferReleases.get());
        assertTrue(api.liveBufferHandles().isEmpty());
    }

    @Test
    void storageTransfersCoverRankZeroAllTypesAndRejectInvalidBoolWithoutPartialPublish() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        MetalBackendRuntime runtime = new MetalBackendRuntime(context);
        try (Arena arena = Arena.ofConfined()) {
            for (DataType type : DataType.values()) {
                int width = type.byteWidth();
                MemorySegment source = arena.allocate(width, width);
                for (int index = 0; index < width; index++) {
                    source.set(JAVA_BYTE, index,
                            type == DataType.BOOL ? (byte) 1 : (byte) (0x10 + index));
                }
                var storage = new MemorySegmentStorage(type, 1L, source);
                TensorDescriptor descriptor = new TensorDescriptor(
                        type,
                        Shape.scalar(),
                        Optional.of(LayoutDescriptor.contiguous(Shape.scalar())),
                        false);
                try (var representation = runtime.borrow(storage)) {
                    assertTrue(runtime.acceptsStorageLayoutTransfer(representation, descriptor));
                    MemorySegment downloaded = arena.allocate(width, width);
                    runtime.bindStorageLayoutDownload(representation, descriptor).accept(downloaded);
                    for (int index = 0; index < width; index++) {
                        assertEquals(source.get(JAVA_BYTE, index),
                                downloaded.get(JAVA_BYTE, index));
                    }
                    byte[] canonical =
                            runtime.copyToCanonicalHostBytes(representation, descriptor, width);
                    for (int index = 0; index < width; index++) {
                        int sourceIndex = type == DataType.BOOL
                                || java.nio.ByteOrder.nativeOrder()
                                        == java.nio.ByteOrder.BIG_ENDIAN
                                ? index : width - 1 - index;
                        assertEquals(source.get(JAVA_BYTE, sourceIndex), canonical[index]);
                    }
                }
            }

            MemorySegment invalidBool = arena.allocate(3L, 1L);
            invalidBool.set(JAVA_BYTE, 0L, (byte) 0);
            invalidBool.set(JAVA_BYTE, 1L, (byte) 2);
            invalidBool.set(JAVA_BYTE, 2L, (byte) 1);
            int uploadsBeforeInvalid = api.uploads.get();
            Shape boolShape = Shape.of(3);
            TensorDescriptor boolDescriptor = new TensorDescriptor(
                    DataType.BOOL,
                    boolShape,
                    Optional.of(LayoutDescriptor.contiguous(boolShape)),
                    false);
            try (var invalidDevice = runtime.borrow(
                    new MemorySegmentStorage(DataType.BOOL, 3L, invalidBool))) {
                MemorySegment untouched = arena.allocate(3L, 1L);
                untouched.fill((byte) 0x5a);
                assertThrows(IllegalArgumentException.class,
                        () -> runtime.bindStorageLayoutDownload(
                                invalidDevice, boolDescriptor).accept(untouched));
                for (int index = 0; index < 3; index++) {
                    assertEquals((byte) 0x5a, untouched.get(JAVA_BYTE, index),
                            "failed BOOL validation must not partially publish");
                }
            }
            assertEquals(uploadsBeforeInvalid + 1, api.uploads.get());

            MemorySegment validBool = arena.allocate(3L, 1L);
            validBool.set(JAVA_BYTE, 0L, (byte) 0);
            validBool.set(JAVA_BYTE, 1L, (byte) 1);
            validBool.set(JAVA_BYTE, 2L, (byte) 0);
            try (var destination = runtime.borrow(
                    new MemorySegmentStorage(DataType.BOOL, 3L, validBool))) {
                assertThrows(IllegalArgumentException.class,
                        () -> runtime.bindStorageLayoutUpload(destination, boolDescriptor)
                                .accept(invalidBool));
                assertEquals(uploadsBeforeInvalid + 2, api.uploads.get());
            }
        } finally {
            runtime.close();
        }
        assertTrue(api.liveBufferHandles().isEmpty());
    }

    @Test
    void firstMetalConsumerOwnsOnePreparedSplatAndCpuSourceOwnsNone() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        BackendPartitionFinalizationResult firstFinalized = null;
        BackendPartitionFinalizationResult secondFinalized = null;
        BackendPartitionFinalizationResult cpuSourceFinalized = null;
        try {
            TensorDescriptor descriptor = descriptor(Shape.of(2));
            ValueId shared = new ValueId(70_000);
            ValueId firstTarget = new ValueId(70_001);
            ValueId secondTarget = new ValueId(70_002);
            CompiledNode firstNode = negNode(70_001, shared, firstTarget);
            CompiledNode secondNode = negNode(70_002, shared, secondTarget);
            PlannedPartition firstPartition = new PlannedPartition(
                    MetalCapabilityProvider.METAL_BACKEND_ID, List.of(firstNode.id()));
            PlannedPartition secondPartition = new PlannedPartition(
                    MetalCapabilityProvider.METAL_BACKEND_ID, List.of(secondNode.id()));
            List<PlannedPartition> consumers = List.of(firstPartition, secondPartition);
            BackendPartitionAnalysis<MetalNegPreparationPlan> first = splatConsumerAnalysis(
                    context, firstPartition, firstNode, shared, firstTarget, descriptor, consumers);
            BackendPartitionAnalysis<MetalNegPreparationPlan> second = splatConsumerAnalysis(
                    context, secondPartition, secondNode, shared, secondTarget, descriptor, consumers);
            assertArrayEquals(new boolean[] {true}, first.plan().feedSplatSources());
            assertArrayEquals(new boolean[] {false}, second.plan().feedSplatSources());

            FinalizationFixture firstFixture = finalization(first);
            firstFinalized = new MetalNegPartitionFinalizer(context)
                    .finalizePartition(firstFixture.finalization());
            FinalizationFixture secondFixture = finalization(second);
            secondFinalized = new MetalNegPartitionFinalizer(context)
                    .finalizePartition(secondFixture.finalization());
            assertEquals(1, api.bufferCreates.get(),
                    "later Metal consumers must not duplicate the source splat");
            assertEquals(1, api.uploads.get());
            assertEquals(2, firstFinalized.resources().size());
            assertEquals(1, secondFinalized.resources().size());

            ValueId cpuOwned = new ValueId(71_000);
            ValueId metalTarget = new ValueId(71_001);
            CompiledNode metalNode = negNode(71_001, cpuOwned, metalTarget);
            PlannedPartition cpuPartition = new PlannedPartition(
                    new io.github.pho001.synaptik.backend.contract.BackendId("cpu-source-test"),
                    List.of(new NodeId(71_000)));
            PlannedPartition metalPartition = new PlannedPartition(
                    MetalCapabilityProvider.METAL_BACKEND_ID, List.of(metalNode.id()));
            BackendPartitionAnalysis<MetalNegPreparationPlan> cpuSource = splatConsumerAnalysis(
                    context,
                    metalPartition,
                    metalNode,
                    cpuOwned,
                    metalTarget,
                    descriptor,
                    List.of(cpuPartition, metalPartition));
            assertArrayEquals(new boolean[] {false}, cpuSource.plan().feedSplatSources());
            FinalizationFixture cpuSourceFixture = finalization(cpuSource);
            cpuSourceFinalized = new MetalNegPartitionFinalizer(context)
                    .finalizePartition(cpuSourceFixture.finalization());
            assertEquals(1, api.bufferCreates.get(),
                    "a CPU source keeps Metal on the ordinary transfer destination path");
            assertEquals(1, cpuSourceFinalized.resources().size());
        } finally {
            closeResources(cpuSourceFinalized);
            closeResources(secondFinalized);
            closeResources(firstFinalized);
            context.close();
        }
        assertTrue(api.liveBufferHandles().isEmpty());
    }

    @Test
    void customPipelineMapsCreateRunReleaseAndMalformedOutputWithoutCrossFamilyCalls() {
        RecordingNativeApi failureApi = new RecordingNativeApi();
        failureApi.pipelineCreateStatus = 12;
        MetalDeviceContext failureContext = MetalDeviceContext.open(failureApi);
        try {
            MetalNegPreparationPlan plan = singleNegRoute(
                    failureContext, Shape.of(2), Optional.empty()).analysis().plan();
            MetalNativeApi.NativeFailure failure = assertThrows(
                    MetalNativeApi.NativeFailure.class,
                    () -> failureContext.createNegKernelPipeline(plan));
            assertNativeFailure(failure,
                    MetalNativeApi.NEG_KERNEL_PIPELINE_CREATE_OPERATION, 12);
            assertEquals(1, failureApi.pipelineCreates.get());
            assertEquals(0, failureApi.pipelineReleases.get());
            assertEquals(0, failureApi.executableCreates.get());
        } finally {
            failureContext.close();
        }

        RecordingNativeApi missingHandleApi = new RecordingNativeApi();
        missingHandleApi.createNullHandle = true;
        MetalDeviceContext missingHandleContext = MetalDeviceContext.open(missingHandleApi);
        try {
            MetalNegPreparationPlan plan = singleNegRoute(
                    missingHandleContext, Shape.of(2), Optional.empty()).analysis().plan();
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> missingHandleContext.createNegKernelPipeline(plan));
            assertTrue(failure.getMessage().contains("returned OK with a null handle"));
            assertEquals(1, missingHandleApi.pipelineCreates.get());
            assertEquals(0, missingHandleApi.pipelineReleases.get());
            assertEquals(0, missingHandleApi.executableCreates.get());
        } finally {
            missingHandleContext.close();
        }

        RecordingNativeApi malformedApi = new RecordingNativeApi();
        malformedApi.pipelineCreateStatus = 12;
        malformedApi.createHandleOnFailure = true;
        malformedApi.pipelineReleaseStatus = 7;
        MetalDeviceContext malformedContext = MetalDeviceContext.open(malformedApi);
        try {
            MetalNegPreparationPlan plan = singleNegRoute(
                    malformedContext, Shape.of(2), Optional.empty()).analysis().plan();
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> malformedContext.createNegKernelPipeline(plan));
            assertTrue(failure.getMessage().contains("failure with a non-null handle"));
            assertNativeFailure((MetalNativeApi.NativeFailure) failure.getCause(),
                    MetalNativeApi.NEG_KERNEL_PIPELINE_CREATE_OPERATION, 12);
            assertEquals(1, failure.getSuppressed().length);
            assertNativeFailure((MetalNativeApi.NativeFailure) failure.getSuppressed()[0],
                    MetalNativeApi.NEG_KERNEL_PIPELINE_CREATE_OPERATION
                            + " malformed-handle cleanup",
                    7);
            assertEquals(1, malformedApi.pipelineReleases.get());
            assertEquals(0, malformedApi.executableReleases.get());
        } finally {
            malformedContext.close();
        }

        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        MetalNegKernelPipelineResource resource = context.createNegKernelPipeline(
                singleNegRoute(context, Shape.of(2), Optional.empty()).analysis().plan());
        MetalBufferRepresentation input = context.createBuffer(8);
        MetalBufferRepresentation output = context.createBuffer(8);
        try {
            api.customRunStatus = 11;
            MetalNativeApi.NativeFailure run = assertThrows(
                    MetalNativeApi.NativeFailure.class,
                    () -> resource.run(input.executionHandle(), output.executionHandle()));
            assertNativeFailure(run, MetalNativeApi.NEG_KERNEL_PIPELINE_RUN_OPERATION, 11);
            assertEquals(1, api.customRunCalls.get());
            assertEquals(0, api.runCalls.get());
        } finally {
            output.close();
            input.close();
            context.close();
        }
        api.pipelineReleaseStatus = 73;
        MetalNativeApi.NativeFailure release = assertThrows(
                MetalNativeApi.NativeFailure.class, resource::close);
        assertNativeFailure(release,
                MetalNativeApi.NEG_KERNEL_PIPELINE_RELEASE_OPERATION, 73);
        resource.close();
        assertEquals(1, api.pipelineReleases.get());
        assertEquals(0, api.executableReleases.get());
    }

    @Test
    void customFinalizerFactoryFailureReversesSplatThenPipelineWithSuppression() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis = singleNegRoute(
                context, Shape.of(2), Optional.of(ScalarValue.float32(3.0f))).analysis();
        FinalizationFixture fixture = finalization(analysis);
        RuntimeException primary = new RuntimeException("custom recipe construction");
        RuntimeException splatCleanup = new RuntimeException("splat cleanup");
        api.bufferReleaseFailures.add(splatCleanup);
        api.pipelineReleaseStatus = 7;
        var finalizer = new MetalNegPartitionFinalizer(context,
                new MetalNegPartitionFinalizer.FinalizedExecutableFactory() {
                    @Override
                    public MetalNegPreparedExecutable createMpsGraph(
                            MetalNegPreparationPlan plan,
                            PreparedMemoryPlan memoryPlan,
                            int[] feeds,
                            int[] feedRepresentations,
                            int[] targets,
                            int[] targetRepresentations,
                            int[] internalPlanIndices,
                            int[] internalRepresentationIndices,
                            MetalMpsGraphExecutableResource resource,
                            List<Optional<MetalPreparedSplatResource>> splats,
                            long[] feedBytes,
                            long[] targetBytes,
                            long[] internalBytes,
                            int workspace) {
                        throw new AssertionError("MPSGraph factory must not be called");
                    }

                    @Override
                    public MetalNegPreparedExecutable createCustom(
                            MetalNegPreparationPlan plan,
                            PreparedMemoryPlan memoryPlan,
                            int feed,
                            int feedRepresentation,
                            int target,
                            int targetRepresentation,
                            MetalNegKernelPipelineResource resource,
                            List<Optional<MetalPreparedSplatResource>> splats) {
                        assertEquals(1, splats.size());
                        assertTrue(splats.getFirst().isPresent());
                        throw primary;
                    }
                });
        try {
            RuntimeException actual = assertThrows(RuntimeException.class,
                    () -> finalizer.finalizePartition(fixture.finalization()));
            assertSame(primary, actual);
            assertEquals(2, actual.getSuppressed().length);
            assertSame(splatCleanup, actual.getSuppressed()[0]);
            assertNativeFailure((MetalNativeApi.NativeFailure) actual.getSuppressed()[1],
                    MetalNativeApi.NEG_KERNEL_PIPELINE_RELEASE_OPERATION, 7);
            assertEquals(1, api.pipelineCreates.get());
            assertEquals(1, api.pipelineReleases.get());
            assertEquals(1, api.bufferCreates.get());
            assertEquals(1, api.uploads.get());
            assertEquals(1, api.bufferReleases.get());
            assertEquals(0, api.executableCreates.get());
            assertTrue(api.liveBufferHandles().isEmpty());
        } finally {
            context.close();
        }
    }

    @Test
    void customColdBindingRejectsAliasWrongContextAndExtentBeforeDowncall() {
        RecordingNativeApi api = new RecordingNativeApi();
        RecordingNativeApi otherApi = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        MetalDeviceContext other = MetalDeviceContext.open(otherApi);
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis = singleNegRoute(
                context, Shape.of(2), Optional.empty()).analysis();
        FinalizationFixture fixture = finalization(analysis);
        BackendPartitionFinalizationResult finalized = new MetalNegPartitionFinalizer(context)
                .finalizePartition(fixture.finalization());
        MetalNegPreparedExecutable executable =
                (MetalNegPreparedExecutable) finalized.executable();
        var good = context.createBuffer(8);
        var otherBuffer = other.createBuffer(8);
        var small = context.createBuffer(4);
        try {
            assertCustomBindingRejected(executable, fixture.memoryPlan(), good, good);
            assertCustomBindingRejected(executable, fixture.memoryPlan(), otherBuffer, good);
            assertCustomBindingRejected(executable, fixture.memoryPlan(), small, good);
            assertEquals(0, api.customRunCalls.get());
        } finally {
            small.close();
            otherBuffer.close();
            good.close();
            finalized.resources().forEach(resource -> resource.close());
            other.close();
            context.close();
        }
    }

    @Test
    void task0052ColdBindingRejectsDistinctLiveValuesSharingOnePhysicalBuffer() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis = task0052MinRoute(context);
        FinalizationFixture fixture = finalization(analysis);
        BackendPartitionFinalizationResult finalized = new MetalNegPartitionFinalizer(context)
                .finalizePartition(fixture.finalization());
        var executable = (MetalNegPreparedExecutable) finalized.executable();
        MetalBufferRepresentation left = context.createBuffer(8);
        MetalBufferRepresentation right = context.createBuffer(8);
        MetalBufferRepresentation targetAlias = new MetalBufferRepresentation(
                context, api, left.executionHandle(), 8);
        RunState state = null;
        try {
            var leftBinding = new BufferRepresentationBinding(
                    left, RunResourceOwnership.BORROWED);
            var rightBinding = new BufferRepresentationBinding(
                    right, RunResourceOwnership.BORROWED);
            var targetBinding = new BufferRepresentationBinding(
                    targetAlias, RunResourceOwnership.BORROWED);
            var workspace = new MetalNegPreparedExecutable.AddressWorkspace(
                    context, analysis.plan().valueIds().size()
                            + analysis.plan().targetValueIds().size());
            state = new RunState(
                    fixture.memoryPlan(),
                    List.of(List.of(leftBinding), List.of(rightBinding), List.of(targetBinding)),
                    List.of(workspace));
            RunState boundState = state;

            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class, () -> executable.bind(boundState));

            assertEquals(
                    "Metal custom-program materialized value buffers must not alias",
                    failure.getMessage());
            assertEquals(0, api.runCalls.get());
        } finally {
            if (state != null) state.close();
            right.close();
            left.close();
            finalized.resources().forEach(resource -> resource.close());
            context.close();
        }
    }

    @Test
    void customPipelineCloseWaitsForAdmittedRunAndReleasesOnce() throws Exception {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        MetalNegKernelPipelineResource resource = context.createNegKernelPipeline(
                singleNegRoute(context, Shape.of(2), Optional.empty()).analysis().plan());
        MetalBufferRepresentation input = context.createBuffer(8);
        MetalBufferRepresentation output = context.createBuffer(8);
        api.blockRuns(1);
        try (var executor = Executors.newFixedThreadPool(2)) {
            var run = executor.submit(
                    () -> resource.run(input.executionHandle(), output.executionHandle()));
            assertTrue(api.runEntered.await(5, TimeUnit.SECONDS));
            context.close();
            var close = executor.submit(resource::close);
            assertEquals(0, api.pipelineReleases.get());
            api.continueRuns.countDown();
            run.get(5, TimeUnit.SECONDS);
            close.get(5, TimeUnit.SECONDS);
        } finally {
            output.close();
            input.close();
            resource.close();
            context.close();
        }
        assertEquals(1, api.customRunCalls.get());
        assertEquals(1, api.pipelineReleases.get());
        assertEquals(1, api.contextReleases.get());
    }

    @Test
    void customPreparedExecutionCloseDefersReleaseAndIsolatesAdmittedRuns()
            throws Exception {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        PreparedExecution execution = prepareSingleExecution(
                context, Shape.of(2), Optional.empty());
        MetalBufferRepresentation caller = context.createBuffer(8);
        uploadBits(caller, 0x3f800000, 0xc0000000);
        long callerHandle = caller.executionHandle().carrier().address();
        int baseCreates = api.bufferCreates.get();
        int baseReleases = api.bufferReleases.get();
        api.blockBufferCreates(2);
        api.blockRuns(1);
        var runner = new PreparedExecutionRunner();
        try (var executor = Executors.newFixedThreadPool(2)) {
            var left = executor.submit(() -> runner.run(execution, List.of(caller)));
            var right = executor.submit(() -> runner.run(execution, List.of(caller)));
            assertTrue(api.bufferCreateEntered.await(5, TimeUnit.SECONDS),
                    "both admitted runs must create their isolated outputs");
            assertTrue(api.runEntered.await(5, TimeUnit.SECONDS));

            execution.close();
            assertTrue(execution.isClosed());
            assertEquals(0, api.pipelineReleases.get(),
                    "admitted runs retain the persistent pipeline after close begins");
            assertEquals(baseCreates + 2, api.bufferCreates.get());
            IllegalStateException rejected = assertThrows(IllegalStateException.class,
                    () -> runner.run(execution, List.of(caller)));
            assertEquals("prepared execution is closed", rejected.getMessage());
            assertEquals(baseCreates + 2, api.bufferCreates.get(),
                    "a rejected run must not create mutable state");

            api.continueRuns.countDown();
            var leftResult = left.get(5, TimeUnit.SECONDS);
            var rightResult = right.get(5, TimeUnit.SECONDS);
            try {
                MetalBufferRepresentation leftOutput =
                        (MetalBufferRepresentation) leftResult.publicationRepresentation(0);
                MetalBufferRepresentation rightOutput =
                        (MetalBufferRepresentation) rightResult.publicationRepresentation(0);
                assertNotSame(leftResult, rightResult);
                assertNotSame(leftOutput, rightOutput);
                assertTrue(leftOutput.executionHandle().carrier().address()
                        != rightOutput.executionHandle().carrier().address());
                assertEquals(2, api.customRunCalls.get());
                assertEquals(1, api.pipelineReleases.get(),
                        "the last admitted run performs deferred pipeline release");
                assertEquals(Set.of(
                        callerHandle,
                        leftOutput.executionHandle().carrier().address(),
                        rightOutput.executionHandle().carrier().address()),
                        api.liveBufferHandles());
            } finally {
                rightResult.close();
                leftResult.close();
            }
            assertEquals(baseReleases + 2, api.bufferReleases.get());
            assertEquals(Set.of(callerHandle), api.liveBufferHandles(),
                    "caller input remains borrowed after both run states close");
        } finally {
            api.continueRuns.countDown();
            execution.close();
            caller.close();
            context.close();
        }
        assertEquals(baseReleases + 3, api.bufferReleases.get());
        assertEquals(1, api.pipelineCreates.get());
        assertEquals(1, api.pipelineReleases.get());
        assertEquals(1, api.contextReleases.get());
        assertTrue(api.liveBufferHandles().isEmpty());
    }

    @Test
    void analyzesTheWholePartitionWithStableValuesFeedsTargetsAndDeclarations() {
        Fixture fixture = fixture();

        MetalNegPreparationPlan plan = analyze(fixture);

        assertEquals(List.of(fixture.v0, fixture.v2, fixture.v3, fixture.v1,
                fixture.v4, fixture.v5), plan.valueIds());
        assertEquals(List.of(
                MetalMpsGraphProgram.NodeKind.NEG,
                MetalMpsGraphProgram.NodeKind.NEG,
                MetalMpsGraphProgram.NodeKind.NEG,
                MetalMpsGraphProgram.NodeKind.NEG),
                plan.graphProgram().nodes().stream()
                        .map(MetalMpsGraphProgram.Node::kind)
                        .toList());
        assertEquals(List.of(fixture.v0, fixture.v1), plan.feedValueIds());
        assertArrayEquals(new int[] {0, 3}, plan.feedValueIndices());
        assertEquals(List.of(fixture.v3, fixture.v4, fixture.v5), plan.targetValueIds());
        assertArrayEquals(new int[] {2, 4, 5}, plan.targetValueIndices());
        assertArrayEquals(new long[] {24, 16}, plan.feedRequiredBytes());
        assertArrayEquals(new long[] {24, 16, 24}, plan.targetRequiredBytes());
        assertEquals(5, plan.declarations().size());
        assertEquals(List.of(fixture.v0, fixture.v1, fixture.v3, fixture.v4, fixture.v5),
                plan.declarations().stream().map(value -> value.valueId()).toList());
    }

    @Test
    void MetalTraceMpsGraphRunObserverErrorIsSuppressedOnNativeFailure() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        AssertionError observerFailure = new AssertionError("MPSGraph run observer");
        MetalTraceProducer producer = new MetalTraceProducer(event -> {
            if (event.payload() instanceof BackendInvocationOutcome) {
                throw observerFailure;
            }
        });
        IndexingRoute route = indexingRoute(context, Map.of(), producer);
        FinalizationFixture finalization = finalization(route.analysis());
        BackendPartitionFinalizationResult finalized =
                new MetalNegPartitionFinalizer(context)
                        .finalizePartition(finalization.finalization());
        var executable = (MetalNegPreparedExecutable) finalized.executable();
        var buffers = new ArrayList<MetalBufferRepresentation>();
        RunState state = null;
        try {
            for (var entry : finalization.memoryPlan().buffers()) {
                buffers.add(context.createBuffer(entry.byteSize()));
            }
            var bindings = buffers.stream()
                    .map(buffer -> List.of(new BufferRepresentationBinding(
                            buffer, RunResourceOwnership.BORROWED)))
                    .toList();
            var workspace = new MetalNegPreparedExecutable.AddressWorkspace(
                    context,
              Math.toIntExact(
                  route.analysis().plan().addressWorkspace().orElseThrow().byteSize()
                      / Long.BYTES));
            state = new RunState(finalization.memoryPlan(), bindings, List.of(workspace));
            var invocation = executable.bind(state);
            api.runStatus = 11;

            MetalNativeApi.NativeFailure failure = assertThrows(
                    MetalNativeApi.NativeFailure.class, invocation::execute);

            assertNativeFailure(failure, MetalNativeApi.EXECUTABLE_RUN_OPERATION, 11);
            assertEquals(
                    MetalNativeApi.EXECUTABLE_RUN_OPERATION
                            + " failed with native status EXECUTION_FAILED (11)",
                    failure.getMessage());
            assertArrayEquals(new Throwable[] {observerFailure}, failure.getSuppressed());
            assertEquals(1, api.runCalls.get());
            assertTrue(producer.enabled());
        } finally {
            if (state != null) state.close();
            buffers.forEach(MetalBufferRepresentation::close);
            finalized.resources().forEach(resource -> resource.close());
            context.close();
        }
    }

    @Test
    void MetalTraceMpsGraphRangeFailureIsCapturedBeforeDeterministicJavaBoundsRescan() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        List<TraceEvent<? extends TracePayload>> events = new CopyOnWriteArrayList<>();
        List<AssertionError> observerFailures = new CopyOnWriteArrayList<>();
        MetalTraceProducer producer = new MetalTraceProducer(event -> {
            events.add(event);
            if (event.payload() instanceof BackendInvocationOutcome) {
                AssertionError failure = new AssertionError("range observer");
                observerFailures.add(failure);
                throw failure;
            }
        });
        IndexingRoute route = indexingRoute(context, Map.of(), producer);
        IndexingRoute splatRoute =
                indexingRoute(context, Map.of(route.oneHotIndices(), ScalarValue.int32(2)));
        MetalNegPreparationPlan plan = route.analysis().plan();
        assertEquals(MetalPreparedRoute.CUSTOM_PROGRAM, plan.route());
        assertEquals(
                List.of(MetalMpsGraphProgram.NodeKind.GATHER,
                        MetalMpsGraphProgram.NodeKind.ONE_HOT),
                plan.graphProgram().nodes().stream()
                        .map(MetalMpsGraphProgram.Node::kind)
                        .toList());
        assertArrayEquals(new long[] {24, 8, 8}, plan.feedRequiredBytes());
        assertArrayEquals(new long[] {16, 8}, plan.targetRequiredBytes());
        assertEquals(List.of(4L, 4L, 4L, 4L, 1L),
                plan.declarations().stream()
                        .map(PreparationResourceRequirement.Buffer::byteAlignment)
                        .toList());
        int splatPosition = splatRoute.analysis().plan().feedValueIds()
                .indexOf(route.oneHotIndices());
        assertEquals(2, splatRoute.analysis().plan().feedSplats()
                .get(splatPosition).orElseThrow().int32Value());

        FinalizationFixture finalization = finalization(route.analysis());
        BackendPartitionFinalizationResult finalized =
                new MetalNegPartitionFinalizer(context)
                        .finalizePartition(finalization.finalization());
        assertEquals(2, events.size());
        assertEquals(TracePhase.PREPARE, events.getLast().phase());
        assertEquals(TraceLevel.INFO, events.getLast().level());
        assertEquals(
                TraceOutcomeStatus.SUCCEEDED,
                ((BackendPreparationOutcome) events.getLast().payload()).status());
        var executable = (MetalNegPreparedExecutable) finalized.executable();
        var buffers = new ArrayList<MetalBufferRepresentation>();
        RunState state = null;
        try {
            for (var entry : finalization.memoryPlan().buffers()) {
                buffers.add(context.createBuffer(entry.byteSize()));
            }
            uploadBits(buffers.get(1), 0, 2);
            uploadBits(buffers.get(2), 1, 4);
            var bindings = buffers.stream()
                    .map(buffer -> List.of(new BufferRepresentationBinding(
                            buffer, RunResourceOwnership.BORROWED)))
                    .toList();
            var workspace = new MetalNegPreparedExecutable.AddressWorkspace(
                    context,
              Math.toIntExact(plan.addressWorkspace().orElseThrow().byteSize() / Long.BYTES));
            state = new RunState(finalization.memoryPlan(), bindings, List.of(workspace));
            var invocation = executable.bind(state);
            api.runStatus = 5;
            IndexOutOfBoundsException oneHot = assertThrows(
                    IndexOutOfBoundsException.class, invocation::execute);
            assertEquals(
                    "ONE_HOT index at logical position 1 is out of bounds: value=4, depth=4",
                    oneHot.getMessage());
            assertArrayEquals(
                    new Throwable[] {observerFailures.get(0)}, oneHot.getSuppressed());

            uploadBits(buffers.get(1), 0, -1);
            uploadBits(buffers.get(2), 1, 2);
            IndexOutOfBoundsException gather = assertThrows(
                    IndexOutOfBoundsException.class, invocation::execute);
            assertEquals(
                    "GATHER index at logical position 1 for data axis 1 is out of bounds:"
                            + " value=-1, extent=3",
                    gather.getMessage());
            assertArrayEquals(
                    new Throwable[] {observerFailures.get(1)}, gather.getSuppressed());
            assertEquals(2, api.runCalls.get());
            assertEquals(6, events.size());
            assertEquals(List.of(0L, 1L, 2L, 3L, 4L, 5L), events.stream()
                    .map(event -> event.id().value())
                    .toList());
            List<TraceEvent<? extends TracePayload>> outcomes = events.stream()
                    .filter(event -> event.payload() instanceof BackendInvocationOutcome)
                    .toList();
            assertEquals(2, outcomes.size());
            for (int index = 0; index < outcomes.size(); index++) {
                TraceEvent<? extends TracePayload> event = outcomes.get(index);
                BackendInvocationOutcome outcome =
                        (BackendInvocationOutcome) event.payload();
                assertEquals(TracePhase.RUN, event.phase());
                assertEquals(TraceLevel.ERROR, event.level());
                assertEquals(TraceOutcomeStatus.FAILED, outcome.status());
                assertEquals(index, outcome.invocationId().value());
                assertEquals(
                        TraceNativeStatusKind.RANGE_OUT_OF_BOUNDS,
                        outcome.nativeStatus().orElseThrow().kind());
                assertEquals(5, outcome.nativeStatus().orElseThrow().code());
            }
            assertTrue(producer.enabled());
        } finally {
            if (state != null) state.close();
            buffers.forEach(MetalBufferRepresentation::close);
            finalized.resources().forEach(resource -> resource.close());
            context.close();
        }
    }

    @Test
    void replacementScatterRescanReproducesExactBoundsAndDuplicateDiagnostics() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        ScatterRoute route = scatterRoute(context);
        FinalizationFixture finalization = finalization(route.analysis());
        BackendPartitionFinalizationResult finalized =
                new MetalNegPartitionFinalizer(context)
                        .finalizePartition(finalization.finalization());
        var executable = (MetalNegPreparedExecutable) finalized.executable();
        var buffers = new ArrayList<MetalBufferRepresentation>();
        RunState state = null;
        try {
            assertEquals(
                    List.of(MetalMpsGraphProgram.NodeKind.SCATTER_ELEMENTS),
                    route.analysis().plan().graphProgram().nodes().stream()
                            .map(MetalMpsGraphProgram.Node::kind)
                            .toList());
            assertArrayEquals(new long[] {24, 16, 16},
                    route.analysis().plan().feedRequiredBytes());
            assertArrayEquals(new long[] {24},
                    route.analysis().plan().targetRequiredBytes());
            for (var entry : finalization.memoryPlan().buffers()) {
                buffers.add(context.createBuffer(entry.byteSize()));
            }
            var bindings = buffers.stream()
                    .map(buffer -> List.of(new BufferRepresentationBinding(
                            buffer, RunResourceOwnership.BORROWED)))
                    .toList();
            var workspace = new MetalNegPreparedExecutable.AddressWorkspace(
                    context,
              Math.toIntExact(
                  route.analysis().plan().addressWorkspace().orElseThrow().byteSize()
                      / Long.BYTES));
            state = new RunState(finalization.memoryPlan(), bindings, List.of(workspace));
            var invocation = executable.bind(state);
            api.runStatus = 5;

            uploadBits(buffers.get(1), 2, 3, 1, 2);
            IndexOutOfBoundsException bounds = assertThrows(
                    IndexOutOfBoundsException.class, invocation::execute);
            assertEquals(
                    "SCATTER_ELEMENTS index at logical position 1 for data axis 1"
                            + " is out of bounds: value=3, extent=3",
                    bounds.getMessage());

            uploadBits(buffers.get(1), 1, 1, 1, 2);
            IllegalArgumentException duplicate = assertThrows(
                    IllegalArgumentException.class, invocation::execute);
            assertEquals(
                    "SCATTER_ELEMENTS duplicate target at logical update position 1;"
                            + " first addressed at logical update position 0",
                    duplicate.getMessage());

            uploadBits(buffers.get(1), 2, 0, 1, 2);
            MetalNativeApi.NativeFailure unmatched = assertThrows(
                    MetalNativeApi.NativeFailure.class, invocation::execute);
            assertEquals(MetalNativeApi.Status.RANGE_OUT_OF_BOUNDS, unmatched.status());
            assertEquals(3, api.runCalls.get());
        } finally {
            if (state != null) state.close();
            buffers.forEach(MetalBufferRepresentation::close);
            finalized.resources().forEach(resource -> resource.close());
            context.close();
        }
    }

    @Test
    void acceleratorBinaryPartitionLowersOrderedBroadcastChainsFanOutAndRepeatedOperands() {
        Fixture fixture = binaryFixture();
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        try {
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                    new MetalNegPartitionPreparer().analyze(new PrepareContext<>(
                            NumericalProfile.ACCELERATOR,
                            new PartitionDag(fixture.partition(), fixture.nodes()),
                            fixture.values(),
                            fixture.requirements(),
                            Map.of(fixture.v1(), ScalarValue.float32(-0.5f)),
                            new MetalNegAnalysisInputs(context)));
            MetalNegPreparationPlan plan = analysis.plan();
            assertEquals(
                    NumericalProfile.ACCELERATOR,
                    plan.numericalProfile());
            assertEquals(MetalPreparedRoute.MPSGRAPH, plan.route());
            assertEquals(List.of(
                    MetalMpsGraphProgram.NodeKind.ADD,
                    MetalMpsGraphProgram.NodeKind.SUB,
                    MetalMpsGraphProgram.NodeKind.MUL,
                    MetalMpsGraphProgram.NodeKind.DIV),
                    plan.graphProgram().nodes().stream()
                            .map(MetalMpsGraphProgram.Node::kind)
                            .toList());
            assertEquals(List.of(fixture.v0(), fixture.v1()), plan.feedValueIds());
            assertArrayEquals(new int[] {0, 1}, plan.feedValueIndices());
            assertEquals(
                    List.of(fixture.v2(), fixture.v3(), fixture.v4(), fixture.v5()),
                    plan.targetValueIds());
            assertArrayEquals(new int[] {2, 3, 4, 5}, plan.targetValueIndices());
            assertTrue(plan.valueStates().stream().allMatch(
                    state -> state == MetalMpsGraphProgram.ValueState.CANONICAL));
            assertEquals(6, plan.declarations().size());
            assertTrue(plan.feedSplats().get(0).isEmpty());
            assertEquals(
                    Float.floatToRawIntBits(-0.5f),
                    Float.floatToRawIntBits(
                            plan.feedSplats().get(1).orElseThrow().float32Value()));

            MetalMpsGraphExecutableResource resource =
                    context.createMpsGraphExecutable(plan);
            try {
                assertEquals(1, api.executableCreates.get());
                assertArrayEquals(
                        plan.graphProgram().encodedProgramImage(
                                plan.numericalProfile(),
                                plan.programValueDescriptors(),
                                plan.feedValueIndices(),
                                plan.targetValueIndices()),
                        api.createdProgramImage);
            } finally {
                resource.close();
            }
        } finally {
            context.close();
        }
    }

    @Test
    void acceleratorMatmulAuthenticatesLocalTransposesAndReusesOnePreparedExecutable() {
        Fixture fixture = matmulFixture();
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> analyze(fixture, context, NumericalProfile.STRICT_IEEE));
            assertEquals(0, api.executableCreates.get());

            TensorDescriptor transposedLeft = descriptorFor(fixture, fixture.v2());
            Fixture boundaryTranspose = withRequirement(
                    fixture,
                    requirement(
                            fixture.v2(),
                            transposedLeft,
                            Optional.of(fixture.partition()),
                            List.of(fixture.partition()),
                            true));
            BackendPartitionAnalysis<MetalNegPreparationPlan> publishedTranspose =
                    analyze(boundaryTranspose, context, NumericalProfile.ACCELERATOR);
            assertTrue(publishedTranspose.plan().targetValueIds().contains(fixture.v2()));

            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                    analyze(fixture, context, NumericalProfile.ACCELERATOR);
            MetalNegPreparationPlan plan = analysis.plan();
            assertEquals(MetalPreparedRoute.CUSTOM_PROGRAM, plan.route());
            assertEquals(List.of(
                    MetalMpsGraphProgram.NodeKind.PERMUTE,
                    MetalMpsGraphProgram.NodeKind.PERMUTE,
                    MetalMpsGraphProgram.NodeKind.MATMUL),
                    plan.graphProgram().nodes().stream()
                            .map(MetalMpsGraphProgram.Node::kind)
                            .toList());
            MetalMpsGraphProgram.Node matmul = plan.graphProgram().nodes().get(2);
            assertEquals(1, matmul.firstInputIndex());
            assertEquals(3, matmul.secondInputIndex());
            assertEquals(4, matmul.outputIndex());
            assertEquals(0, matmul.attributeCount());
            assertEquals(MetalMpsGraphProgram.NO_AXIS, matmul.axis());
            assertEquals(List.of(fixture.v0(), fixture.v1()), plan.feedValueIds());
            assertEquals(List.of(fixture.v4()), plan.targetValueIds());
      assertEquals(List.of(fixture.v2(), fixture.v3()),plan.internalValueIds());
            assertEquals(List.of(
                    MetalMpsGraphProgram.ValueState.CANONICAL,
                    MetalMpsGraphProgram.ValueState.AFFINE_VIEW,
                    MetalMpsGraphProgram.ValueState.CANONICAL,
                    MetalMpsGraphProgram.ValueState.AFFINE_VIEW,
                    MetalMpsGraphProgram.ValueState.CANONICAL),
                    plan.valueStates());

            try (var left = context.createBuffer(24);
                    var leftViewBuffer = context.createBuffer(24);
          var right = context.createBuffer(48);
          var rightViewBuffer = context.createBuffer(48);
                    var output = context.createBuffer(32);
                    var workspace = addressWorkspace(context, left, leftViewBuffer, right, rightViewBuffer, output, output);
                    var resource = context.createMpsGraphExecutable(plan)) {
                resource.run(
            5,
                        workspace.segment().asSlice(0, 5L * Long.BYTES),
                        1,
                        workspace.segment().asSlice(5L * Long.BYTES, Long.BYTES));
                resource.run(
            5,
                        workspace.segment().asSlice(0, 5L * Long.BYTES),
                        1,
                        workspace.segment().asSlice(5L * Long.BYTES, Long.BYTES));
                assertEquals(1, api.executableCreates.get());
                assertEquals(2, api.runCalls.get());
                assertArrayEquals(
                        plan.graphProgram().encodedProgramImage(
                                plan.numericalProfile(),
                                plan.programValueDescriptors(),
                                plan.feedValueIndices(),
                                plan.targetValueIndices(),
                                plan.route()),
                        api.createdProgramImage);
            }
            assertEquals(1, api.executableReleases.get());
        } finally {
            context.close();
        }
        assertEquals(1, api.contextReleases.get());
    }
    @Test
    void generalGeometryMatmulSelectsCustomProgramBeforeResourceDeclaration() {
        Fixture fixture = batchedMatmulFixture();
        RecordingNativeApi api = new RecordingNativeApi();
        try (MetalDeviceContext context = MetalDeviceContext.open(api)) {
            assertThrows(IllegalArgumentException.class,
                    () -> analyze(fixture, context, NumericalProfile.STRICT_IEEE));
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                    analyze(fixture, context, NumericalProfile.ACCELERATOR);
            MetalNegPreparationPlan plan = analysis.plan();
            assertEquals(MetalPreparedRoute.CUSTOM_PROGRAM, plan.route());
            assertEquals(
                    List.of(MetalMpsGraphProgram.NodeKind.MATMUL),
                    plan.graphProgram().nodes().stream()
                            .map(MetalMpsGraphProgram.Node::kind)
                            .toList());
            assertEquals(List.of(fixture.v0(), fixture.v1()), plan.feedValueIds());
            assertEquals(List.of(fixture.v2()), plan.targetValueIds());
            assertTrue(plan.internalValueIds().isEmpty());
            assertEquals(
                    List.of(fixture.v0(), fixture.v1(), fixture.v2()),
                    analysis.requirements().stream()
                            .filter(PreparationResourceRequirement.Buffer.class::isInstance)
                            .map(PreparationResourceRequirement.Buffer.class::cast)
                            .map(PreparationResourceRequirement.Buffer::valueId)
                            .toList());
        }
        assertEquals(1, api.contextReleases.get());
    }


    @Test
    void acceleratorReductionsLowerTypedFormsAndReuseOnePreparedExecutable() {
        Fixture fixture = reductionFixture();
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> analyze(fixture, context, NumericalProfile.STRICT_IEEE));
            assertEquals(0, api.executableCreates.get());

            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                    analyze(fixture, context, NumericalProfile.ACCELERATOR);
            MetalNegPreparationPlan plan = analysis.plan();
            assertEquals(NumericalProfile.ACCELERATOR, plan.numericalProfile());
            assertEquals(MetalPreparedRoute.MPSGRAPH, plan.route());
            assertEquals(List.of(
                    MetalMpsGraphProgram.NodeKind.MEAN,
                    MetalMpsGraphProgram.NodeKind.ABS,
                    MetalMpsGraphProgram.NodeKind.SUM,
                    MetalMpsGraphProgram.NodeKind.SUM,
                    MetalMpsGraphProgram.NodeKind.SUM),
                    plan.graphProgram().nodes().stream()
                            .map(MetalMpsGraphProgram.Node::kind)
                            .toList());
            List<MetalMpsGraphProgram.Node> nodes = plan.graphProgram().nodes();
            assertTypedNode(nodes.get(0), MetalMpsGraphProgram.NodeKind.MEAN,
                    0, 1, 1,
                    MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS.wireIdentity(), 1L);
            assertTypedNode(nodes.get(1), MetalMpsGraphProgram.NodeKind.ABS,
                    1, 2, 0, MetalMpsGraphProgram.NO_AXIS);
            assertTypedNode(nodes.get(2), MetalMpsGraphProgram.NodeKind.SUM,
                    0, 3, 0,
                    MetalMpsGraphProgram.ReductionForm.FULL.wireIdentity());
            assertTypedNode(nodes.get(3), MetalMpsGraphProgram.NodeKind.SUM,
                    0, 4, 2,
                    MetalMpsGraphProgram.ReductionForm.SUM_TO_SHAPE.wireIdentity(), 1L, 4L);
            assertTypedNode(nodes.get(4), MetalMpsGraphProgram.NodeKind.SUM,
                    0, 5, 0,
                    MetalMpsGraphProgram.ReductionForm.MULTI_AXIS.wireIdentity());
            assertEquals(List.of(fixture.v0()), plan.feedValueIds());
            assertArrayEquals(new int[] {0}, plan.feedValueIndices());
            assertEquals(
                    List.of(fixture.v2(), fixture.v3(), fixture.v4(), fixture.v5()),
                    plan.targetValueIds());
            assertArrayEquals(new int[] {2, 3, 4, 5}, plan.targetValueIndices());
            assertArrayEquals(new long[] {96}, plan.feedRequiredBytes());
            assertArrayEquals(new long[] {32, 4, 16, 96}, plan.targetRequiredBytes());
            assertEquals(5, plan.declarations().size());
            assertTrue(plan.valueStates().stream().allMatch(
                    state -> state == MetalMpsGraphProgram.ValueState.CANONICAL));

            try (var input = context.createBuffer(96);
                    var meanAbs = context.createBuffer(32);
                    var scalarSum = context.createBuffer(4);
                    var sumTo = context.createBuffer(16);
                    var identity = context.createBuffer(96);
                    var workspace = addressWorkspace(
                            context, input, meanAbs, scalarSum, sumTo, identity);
                    var resource = context.createMpsGraphExecutable(plan)) {
                resource.run(1, workspace.segment().asSlice(0, Long.BYTES),
                        4, workspace.segment().asSlice(Long.BYTES, 4L * Long.BYTES));
                resource.run(1, workspace.segment().asSlice(0, Long.BYTES),
                        4, workspace.segment().asSlice(Long.BYTES, 4L * Long.BYTES));
                assertEquals(1, api.executableCreates.get());
                assertEquals(2, api.runCalls.get());
                assertArrayEquals(
                        plan.graphProgram().encodedProgramImage(
                                plan.numericalProfile(),
                                plan.programValueDescriptors(),
                                plan.feedValueIndices(),
                                plan.targetValueIndices()),
                        api.createdProgramImage);
            }
            assertEquals(1, api.executableReleases.get());
        } finally {
            context.close();
        }
        assertEquals(1, api.contextReleases.get());
    }

    @Test
    void acceleratorMseLowersEveryReductionToOneReusableMpsGraphExecutable() {
        Fixture fixture = mseFixture();
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> analyze(fixture, context, NumericalProfile.STRICT_IEEE));
            assertEquals(0, api.executableCreates.get());

            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                    analyze(fixture, context, NumericalProfile.ACCELERATOR);
            MetalNegPreparationPlan plan = analysis.plan();
            assertEquals(MetalPreparedRoute.MPSGRAPH, plan.route());
            assertEquals(
                    List.of(
                            MetalMpsGraphProgram.NodeKind.MEAN_SQUARED_ERROR,
                            MetalMpsGraphProgram.NodeKind.MEAN_SQUARED_ERROR,
                            MetalMpsGraphProgram.NodeKind.MEAN_SQUARED_ERROR),
                    plan.graphProgram().nodes().stream()
                            .map(MetalMpsGraphProgram.Node::kind)
                            .toList());
            for (int index = 0; index < 3; index++) {
                MetalMpsGraphProgram.Node node = plan.graphProgram().nodes().get(index);
                assertArrayEquals(new int[] {0, 1}, node.inputs());
                assertArrayEquals(new int[] {index + 2}, node.outputs());
                assertEquals(MetalMpsGraphProgram.AttributeKind.MSE, node.attributeKind());
                assertArrayEquals(new long[] {index + 1L}, node.attributeWords());
            }
            assertEquals(List.of(fixture.v0(), fixture.v1()), plan.feedValueIds());
            assertArrayEquals(new int[] {0, 1}, plan.feedValueIndices());
            assertEquals(
                    List.of(fixture.v2(), fixture.v3(), fixture.v4()),
                    plan.targetValueIds());
            assertArrayEquals(new int[] {2, 3, 4}, plan.targetValueIndices());
            assertTrue(plan.internalValueIds().isEmpty());
            assertArrayEquals(new long[] {24, 24}, plan.feedRequiredBytes());
            assertArrayEquals(new long[] {24, 4, 4}, plan.targetRequiredBytes());
            assertEquals(5, plan.declarations().size());

            try (var prediction = context.createBuffer(24);
                    var target = context.createBuffer(24);
                    var none = context.createBuffer(24);
                    var sum = context.createBuffer(4);
                    var mean = context.createBuffer(4);
                    var workspace =
                            addressWorkspace(context, prediction, target, none, sum, mean);
                    var resource = context.createMpsGraphExecutable(plan)) {
                resource.run(
                        2,
                        workspace.segment().asSlice(0, 2L * Long.BYTES),
                        3,
                        workspace.segment().asSlice(2L * Long.BYTES, 3L * Long.BYTES));
                resource.run(
                        2,
                        workspace.segment().asSlice(0, 2L * Long.BYTES),
                        3,
                        workspace.segment().asSlice(2L * Long.BYTES, 3L * Long.BYTES));
                assertEquals(1, api.executableCreates.get());
                assertEquals(2, api.runCalls.get());
            }
            assertEquals(1, api.executableReleases.get());
        } finally {
            context.close();
        }
        assertEquals(1, api.contextReleases.get());
    }

    @Test
    void compilesOnceRunsInStableOrderAndDefersContextReleaseUntilResourceClose() {
        Fixture fixture = fixture();
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        MetalMpsGraphExecutableResource resource =
                context.createMpsGraphExecutable(analyze(fixture, context).plan());
        var first = context.createBuffer(24);
        var second = context.createBuffer(16);
        var out0 = context.createBuffer(24);
        var out1 = context.createBuffer(16);
        var out2 = context.createBuffer(24);

        var workspace = addressWorkspace(context, first, second, out0, out1, out2);
        resource.run(2, workspace.segment().asSlice(0, 16),
                3, workspace.segment().asSlice(16, 24));
        resource.run(2, workspace.segment().asSlice(0, 16),
                3, workspace.segment().asSlice(16, 24));

        assertEquals(1, api.executableCreates.get());
        assertEquals(2, api.runCalls.get());
        context.close();
        assertEquals(0, api.contextReleases.get());
        workspace.close(); out2.close(); out1.close(); out0.close(); second.close(); first.close();
        assertEquals(0, api.contextReleases.get());
        resource.close();
        resource.close();
        assertEquals(1, api.executableReleases.get());
        assertEquals(1, api.contextReleases.get());
        assertEquals(1, api.apiCloses.get());
    }

    @Test
    void nativeCompilationFailureRollsBackTheProvisionalLease() {
        RecordingNativeApi api = new RecordingNativeApi();
        RuntimeException expected = new RuntimeException("compile");
        api.createFailure = expected;
        MetalDeviceContext context = MetalDeviceContext.open(api);

        assertSame(expected, assertThrows(RuntimeException.class,
                () -> context.createMpsGraphExecutable(analyze(fixture(), context).plan())));
        context.close();

        assertEquals(0, api.executableReleases.get());
        assertEquals(1, api.contextReleases.get());
        assertEquals(1, api.apiCloses.get());
    }

    @Test
    void contextRejectsForeignPreparationPlanBeforeLeaseOrNativeCreate() {
        RecordingNativeApi firstApi = new RecordingNativeApi();
        RecordingNativeApi secondApi = new RecordingNativeApi();
        MetalDeviceContext first = MetalDeviceContext.open(firstApi);
        MetalDeviceContext second = MetalDeviceContext.open(secondApi);
        MetalNegPreparationPlan plan = analyze(fixture(), first).plan();
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> second.createMpsGraphExecutable(plan));
            assertEquals(0, secondApi.executableCreates.get());
            second.close();
            assertEquals(1, secondApi.contextReleases.get(),
                    "context mismatch must not leave a provisional child lease");
        } finally {
            second.close();
            first.close();
        }
    }

    @Test
    void finalizerRejectsRepeatedBufferAssignmentsAcrossEveryRoleBeforeAcquisition() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                analyze(fixture(Shape.of(2, 3)), context);
        FinalizationFixture valid = finalization(analysis);
        int feedCount = analysis.plan().feedValueIds().size();
        int[][] duplicatePairs = {
                {0, 1},
                {feedCount, feedCount + 1},
                {0, feedCount}
        };
        try {
            for (int[] pair : duplicatePairs) {
                var assignments = new ArrayList<>(valid.assignments());
                var source = (PreparationResourceAssignment.Buffer) assignments.get(pair[0]);
                var replaced = (PreparationResourceAssignment.Buffer) assignments.get(pair[1]);
                assignments.set(pair[1], new PreparationResourceAssignment.Buffer(replaced.requirement(), source.slot(), source.planIndex(), 0));
                var malformed = new BackendPartitionFinalization<>(
                        analysis, valid.memoryPlan(), assignments);
                assertThrows(IllegalArgumentException.class,
                        () -> new MetalNegPartitionFinalizer(context)
                                .finalizePartition(malformed));
                assertEquals(0, api.executableCreates.get());
            }
        } finally {
            context.close();
        }
    }

    @Test
    void analysisRequiresExactLogicalProducerAndConsumerPartitionFacts() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        Fixture base = fixture();
        PlannedPartition copy = new PlannedPartition(
                base.partition().owner(), base.partition().nodeIds());
        TensorDescriptor a = descriptorFor(base, base.v0());
        try {
            List<LogicalMemoryRequirement> invalid = List.of(
                    requirement(base.v2(), a, Optional.of(copy),
                            List.of(base.partition()), false),
                    requirement(base.v0(), a, Optional.empty(), List.of(copy), false),
                    requirement(base.v0(), a, Optional.of(base.partition()),
                            List.of(base.partition()), false),
                    requirement(base.v2(), a, Optional.empty(),
                            List.of(base.partition()), false),
                    requirement(base.v0(), a, Optional.empty(), List.of(), false),
                    requirement(base.v3(), a, Optional.of(base.partition()),
                            List.of(), true));
            for (LogicalMemoryRequirement replacement : invalid) {
                Fixture malformed = withRequirement(base, replacement);
                assertThrows(IllegalArgumentException.class,
                        () -> analyze(malformed, context));
                assertEquals(0, api.executableCreates.get());
            }

            PlannedPartition outside = new PlannedPartition(
                    MetalCapabilityProvider.METAL_BACKEND_ID, List.of(new NodeId(9999)));
            Fixture legitimateOutside = withRequirement(base,
                    requirement(base.v2(), a, Optional.of(base.partition()),
                            List.of(base.partition(), outside), false));
            assertTrue(analyze(legitimateOutside, context).plan()
                    .targetValueIds().contains(base.v2()));
        } finally {
            context.close();
        }
    }

    @Test
    void executableCreateMapsStatusesAndMalformedOutputCellsWithoutRetry() {
        for (int status : new int[] {9, 7, 73}) {
            RecordingNativeApi api = new RecordingNativeApi();
            api.createStatus = status;
            MetalDeviceContext context = MetalDeviceContext.open(api);
            try {
                MetalNativeApi.NativeFailure failure = assertThrows(
                        MetalNativeApi.NativeFailure.class,
                        () -> context.createMpsGraphExecutable(analyze(fixture(), context).plan()));
                assertNativeFailure(failure,
                        MetalNativeApi.EXECUTABLE_CREATE_OPERATION, status);
                assertEquals(1, api.executableCreates.get());
                assertEquals(0, api.executableReleases.get());
            } finally {
                context.close();
            }
            assertEquals(1, api.contextReleases.get(),
                    "failed create must release its provisional context lease");
        }

        RecordingNativeApi nullApi = new RecordingNativeApi();
        nullApi.createNullHandle = true;
        MetalDeviceContext nullContext = MetalDeviceContext.open(nullApi);
        try {
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> nullContext.createMpsGraphExecutable(analyze(fixture(), nullContext).plan()));
            assertTrue(failure.getMessage().contains("returned OK with a null handle"));
            assertEquals(1, nullApi.executableCreates.get());
            assertEquals(0, nullApi.executableReleases.get());
        } finally {
            nullContext.close();
        }

        RecordingNativeApi malformedApi = new RecordingNativeApi();
        malformedApi.createStatus = 9;
        malformedApi.createHandleOnFailure = true;
        malformedApi.releaseStatus = 7;
        MetalDeviceContext malformedContext = MetalDeviceContext.open(malformedApi);
        try {
            IllegalStateException failure = assertThrows(IllegalStateException.class,
                    () -> malformedContext.createMpsGraphExecutable(analyze(fixture(), malformedContext).plan()));
            assertTrue(failure.getMessage().contains("failure with a non-null handle"));
            assertNativeFailure((MetalNativeApi.NativeFailure) failure.getCause(),
                    MetalNativeApi.EXECUTABLE_CREATE_OPERATION, 9);
            assertEquals(1, failure.getSuppressed().length);
            assertNativeFailure((MetalNativeApi.NativeFailure) failure.getSuppressed()[0],
                    MetalNativeApi.EXECUTABLE_CREATE_OPERATION
                            + " malformed-handle cleanup",
                    7);
            assertEquals(1, malformedApi.executableCreates.get());
            assertEquals(1, malformedApi.executableReleases.get());
        } finally {
            malformedContext.close();
        }
    }

    @Test
    void executableRunAndReleaseMapNativeStatusesWithoutRetryOrFallback() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        MetalMpsGraphExecutableResource resource =
                context.createMpsGraphExecutable(analyze(fixture(), context).plan());
        var inputs = List.of(context.createBuffer(24), context.createBuffer(16));
        var outputs = List.of(context.createBuffer(24), context.createBuffer(16),
                context.createBuffer(24));
        var workspace = addressWorkspace(context,
                inputs.get(0), inputs.get(1), outputs.get(0), outputs.get(1), outputs.get(2));
        try {
            int priorCalls = api.runCalls.get();
            for (int status : new int[] {10, 11, 7, 79}) {
                api.runStatus = status;
                MetalNativeApi.NativeFailure failure = assertThrows(
                        MetalNativeApi.NativeFailure.class,
                        () -> resource.run(2, workspace.segment().asSlice(0, 16),
                                3, workspace.segment().asSlice(16, 24)));
                assertNativeFailure(failure, MetalNativeApi.EXECUTABLE_RUN_OPERATION, status);
                assertEquals(++priorCalls, api.runCalls.get());
            }
            api.runStatus = 0;
        } finally {
            workspace.close();
            outputs.forEach(MetalBufferRepresentation::close);
            inputs.forEach(MetalBufferRepresentation::close);
        }

        context.close();
        api.releaseStatus = 7;
        MetalNativeApi.NativeFailure release = assertThrows(
                MetalNativeApi.NativeFailure.class, resource::close);
        assertNativeFailure(release, MetalNativeApi.EXECUTABLE_RELEASE_OPERATION, 7);
        resource.close();
        assertEquals(1, api.executableReleases.get());
        assertEquals(1, api.contextReleases.get());

        RecordingNativeApi unknownApi = new RecordingNativeApi();
        MetalDeviceContext unknownContext = MetalDeviceContext.open(unknownApi);
        MetalMpsGraphExecutableResource unknownResource =
                unknownContext.createMpsGraphExecutable(analyze(fixture(), unknownContext).plan());
        unknownContext.close();
        unknownApi.releaseStatus = 83;
        MetalNativeApi.NativeFailure unknownRelease = assertThrows(
                MetalNativeApi.NativeFailure.class, unknownResource::close);
        assertNativeFailure(
                unknownRelease, MetalNativeApi.EXECUTABLE_RELEASE_OPERATION, 83);
        unknownResource.close();
        assertEquals(1, unknownApi.executableReleases.get());
        assertEquals(1, unknownApi.contextReleases.get());
    }

    @Test
    void executableCleanupPreservesReleasePrimaryThenContextAndApiFailures() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        MetalMpsGraphExecutableResource resource =
                context.createMpsGraphExecutable(analyze(fixture(), context).plan());
        RuntimeException contextFailure = new RuntimeException("context release");
        RuntimeException closeFailure = new RuntimeException("api close");
        api.releaseStatus = 7;
        api.contextReleaseFailure = contextFailure;
        api.apiCloseFailure = closeFailure;

        context.close();
        MetalNativeApi.NativeFailure failure = assertThrows(
                MetalNativeApi.NativeFailure.class, resource::close);

        assertNativeFailure(failure, MetalNativeApi.EXECUTABLE_RELEASE_OPERATION, 7);
        assertArrayEquals(new Throwable[] {contextFailure, closeFailure}, failure.getSuppressed());
        resource.close();
        assertEquals(1, api.executableReleases.get());
        assertEquals(1, api.contextReleases.get());
        assertEquals(1, api.apiCloses.get());
    }

    @Test
    void executableCloseWaitsForRunAndConcurrentRunsReuseOneResourceWithoutRetry()
            throws Exception {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        MetalMpsGraphExecutableResource resource =
                context.createMpsGraphExecutable(analyze(fixture(), context).plan());
        var first = context.createBuffer(24);
        var second = context.createBuffer(16);
        var out0 = context.createBuffer(24);
        var out1 = context.createBuffer(16);
        var out2 = context.createBuffer(24);
        var workspace = addressWorkspace(context, first, second, out0, out1, out2);
        try (var executor = Executors.newFixedThreadPool(3)) {
            api.blockRuns(1);
            var firstRun = executor.submit(() -> resource.run(
                    2, workspace.segment().asSlice(0, 16),
                    3, workspace.segment().asSlice(16, 24)));
            assertTrue(api.runEntered.await(5, TimeUnit.SECONDS));
            CountDownLatch secondStarted = new CountDownLatch(1);
            var secondRun = executor.submit(() -> {
                secondStarted.countDown();
                resource.run(2, workspace.segment().asSlice(0, 16),
                        3, workspace.segment().asSlice(16, 24));
            });
            assertTrue(secondStarted.await(5, TimeUnit.SECONDS));
            assertEquals(1, api.runCalls.get(),
                    "the reusable resource serializes its synchronous native boundary");
            CountDownLatch closeStarted = new CountDownLatch(1);
            var close = executor.submit(() -> {
                closeStarted.countDown();
                resource.close();
            });
            assertTrue(closeStarted.await(5, TimeUnit.SECONDS));
            assertEquals(0, api.executableReleases.get(),
                    "close must not release while an admitted run holds the resource gate");
            api.continueRuns.countDown();
            firstRun.get(5, TimeUnit.SECONDS);
            secondRun.get(5, TimeUnit.SECONDS);
            close.get(5, TimeUnit.SECONDS);
        } finally {
            api.continueRuns.countDown();
            workspace.close();
            out2.close(); out1.close(); out0.close(); second.close(); first.close();
            resource.close();
            context.close();
        }
        assertEquals(2, api.runCalls.get());
        assertEquals(1, api.executableReleases.get());
    }

    @Test
    void finalizerRollsBackAcquiredExecutableAndPreservesPrimaryCleanupSuppression() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        Fixture fixture = fixture();
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis = analyze(fixture, context);
        FinalizationFixture finalization = finalization(analysis);
        RuntimeException primary = new RuntimeException("recipe construction");
        api.releaseStatus = 7;
        var finalizer = new MetalNegPartitionFinalizer(context,
                (plan,
                        memoryPlan,
                        feeds,
                        feedRepresentations,
                        targets,
                        targetRepresentations,
                        internalPlanIndices,
                        internalRepresentationIndices,
                        resource,
                        splats,
                        feedBytes,
                        targetBytes,
                        internalBytes,
                        workspace) -> {
                    throw primary;
                });
        try {
            RuntimeException failure = assertThrows(RuntimeException.class,
                    () -> finalizer.finalizePartition(finalization.finalization()));
            assertSame(primary, failure);
            assertEquals(1, failure.getSuppressed().length);
            assertNativeFailure((MetalNativeApi.NativeFailure) failure.getSuppressed()[0],
                    MetalNativeApi.EXECUTABLE_RELEASE_OPERATION, 7);
            assertEquals(1, api.executableCreates.get());
            assertEquals(1, api.executableReleases.get());
        } finally {
            context.close();
        }
    }

    @Test
    void finalizerAndAssemblerRejectContextOrPartitionMismatchBeforeAcquisitionOrCreators() {
        RecordingNativeApi firstApi = new RecordingNativeApi();
        RecordingNativeApi secondApi = new RecordingNativeApi();
        MetalDeviceContext first = MetalDeviceContext.open(firstApi);
        MetalDeviceContext second = MetalDeviceContext.open(secondApi);
        Fixture fixture = fixture();
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis = analyze(fixture, first);
        FinalizationFixture prepared = finalization(analysis);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> new MetalNegPartitionFinalizer(second)
                            .finalizePartition(prepared.finalization()));
            assertEquals(0, firstApi.executableCreates.get());
            assertEquals(0, secondApi.executableCreates.get());

            PlannedPartition otherPartition = new PlannedPartition(
                    MetalCapabilityProvider.METAL_BACKEND_ID,
                    fixture.partition().nodeIds());
            BackendPartitionAnalysis<MetalNegPreparationPlan> mismatched =
                    new BackendPartitionAnalysis<>(otherPartition, analysis.plan(),
                            analysis.requirements());
            assertThrows(IllegalArgumentException.class,
                    () -> new MetalNegPartitionFinalizer(first).finalizePartition(
                            new BackendPartitionFinalization<>(mismatched,
                                    prepared.memoryPlan(), prepared.assignments())));
            assertEquals(0, firstApi.executableCreates.get());

            assertThrows(IllegalArgumentException.class,
                    () -> new MetalNegPreparedScheduleAssembler(
                            second, analysis.plan(), List.of(fixture.v3())));
            assertEquals(0, secondApi.bufferCreates.get());

            BackendPartitionFinalizationResult finalized =
                    new MetalNegPartitionFinalizer(first)
                            .finalizePartition(prepared.finalization());
            try {
                int beforeBuffers = firstApi.bufferCreates.get();
                assertThrows(IllegalArgumentException.class,
                        () -> new MetalNegPreparedScheduleAssembler(
                                first, analysis.plan(), analysis.plan().targetValueIds())
                                .assembleRoute(prepared.memoryPlan(),
                                        new PreparedPartition(
                                                otherPartition, finalized.executable()),
                                        prepared.preparedAssignments()));
                assertEquals(beforeBuffers, firstApi.bufferCreates.get());
            } finally {
                finalized.resources().forEach(resource -> resource.close());
            }
        } finally {
            second.close();
            first.close();
        }
    }

    @Test
    void bindingRejectsWrongClosedUndersizedAndWorkspaceRepresentationsBeforeDowncall() {
        RecordingNativeApi api = new RecordingNativeApi();
        RecordingNativeApi otherApi = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        MetalDeviceContext other = MetalDeviceContext.open(otherApi);
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis = analyze(fixture(), context);
        FinalizationFixture fixture = finalization(analysis);
        BackendPartitionFinalizationResult finalized =
                new MetalNegPartitionFinalizer(context)
                        .finalizePartition(fixture.finalization());
        MetalNegPreparedExecutable executable =
                (MetalNegPreparedExecutable) finalized.executable();
        try {
            assertBindingRejected(executable, fixture.memoryPlan(), context, other,
                    BindingDefect.WRONG_BUFFER_CONTEXT, api);
            assertBindingRejected(executable, fixture.memoryPlan(), context, other,
                    BindingDefect.CLOSED_BUFFER, api);
            assertBindingRejected(executable, fixture.memoryPlan(), context, other,
                    BindingDefect.UNDERSIZED_BUFFER, api);
            assertBindingRejected(executable, fixture.memoryPlan(), context, other,
                    BindingDefect.OUTPUT_ALIAS, api);
            assertBindingRejected(executable, fixture.memoryPlan(), context, other,
                    BindingDefect.WRONG_WORKSPACE_CONTEXT, api);
            assertBindingRejected(executable, fixture.memoryPlan(), context, other,
                    BindingDefect.WRONG_WORKSPACE_COUNT, api);
            assertBindingRejected(executable, fixture.memoryPlan(), context, other,
                    BindingDefect.CLOSED_WORKSPACE, api);
            assertEquals(0, api.runCalls.get());
        } finally {
            finalized.resources().forEach(resource -> resource.close());
            other.close();
            context.close();
        }
    }

    @Test
    void JavaRunPreflightRejectsAliasAndAddressGeometryBeforeNativeDowncall() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        MetalMpsGraphExecutableResource resource =
                context.createMpsGraphExecutable(analyze(fixture(), context).plan());
        var input0 = context.createBuffer(24);
        var input1 = context.createBuffer(16);
        var output0 = context.createBuffer(24);
        var output1 = context.createBuffer(16);
        var output2 = context.createBuffer(24);
        var workspace = new MetalNegPreparedExecutable.AddressWorkspace(context, 5);
        try {
            workspace.set(0, input0.executionHandle());
            workspace.set(1, input1.executionHandle());
            workspace.set(2, output0.executionHandle());
            workspace.set(3, output0.executionHandle());
            workspace.set(4, output2.executionHandle());
            assertThrows(IllegalArgumentException.class,
                    () -> resource.run(2, workspace.segment().asSlice(0, 16),
                            3, workspace.segment().asSlice(16, 24)));
            assertEquals(0, api.runCalls.get());

            workspace.set(2, input0.executionHandle());
            workspace.set(3, output1.executionHandle());
            assertThrows(IllegalArgumentException.class,
                    () -> resource.run(2, workspace.segment().asSlice(0, 16),
                            3, workspace.segment().asSlice(16, 24)));
            assertEquals(0, api.runCalls.get());

            assertThrows(IllegalArgumentException.class,
                    () -> resource.run(1, workspace.segment().asSlice(0, 8),
                            3, workspace.segment().asSlice(16, 24)));
            assertThrows(IllegalArgumentException.class,
                    () -> resource.run(2, workspace.segment().asSlice(0, 8),
                            3, workspace.segment().asSlice(16, 24)));
            assertEquals(0, api.runCalls.get());
        } finally {
            workspace.close();
            output2.close(); output1.close(); output0.close(); input1.close(); input0.close();
            resource.close();
            context.close();
        }
    }

    @Test
    void nontrivialPlanIndicesDriveExecutableSelectionsAndRepresentationPreparations() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis = analyze(fixture(), context);
        FinalizationFixture reversed = finalization(analysis, true);
        BackendPartitionFinalizationResult finalized =
                new MetalNegPartitionFinalizer(context)
                        .finalizePartition(reversed.finalization());
        try {
            MetalNegPreparedExecutable executable =
                    (MetalNegPreparedExecutable) finalized.executable();
            int declarationCount = analysis.plan().declarations().size();
            for (int selection = 0; selection < declarationCount; selection++) {
                assertEquals(declarationCount - 1 - selection,
                        executable.bufferSelection(selection).bufferIndex());
            }
            var schedule = new MetalNegPreparedScheduleAssembler(
                    context, analysis.plan(), analysis.plan().targetValueIds())
                    .assembleRoute(reversed.memoryPlan(),
                            new PreparedPartition(analysis.partition(), executable),
                            reversed.preparedAssignments());
            var creation = (io.github.pho001.synaptik.runtime.schedule.PreparedSchedule
                    .RepresentationCreationStep) schedule.steps().getFirst();
            for (int planIndex = 0; planIndex < declarationCount; planIndex++) {
                ValueId value = reversed.preparedAssignments().get(planIndex).valueId();
                Object preparation = creation.representationPlan()
                        .bufferPreparations().get(planIndex).getFirst();
                if (analysis.plan().feedValueIds().contains(value)) {
                    assertTrue(preparation
                            instanceof io.github.pho001.synaptik.runtime.resource
                                    .PreparedRepresentationPlan.CallerInput);
                } else {
                    assertTrue(preparation
                            instanceof io.github.pho001.synaptik.runtime.resource
                                    .PreparedRepresentationPlan.CreatedBuffer);
                }
            }
            assertEquals(0, api.bufferCreates.get(),
                    "assembly must not invoke any reordered creator");
        } finally {
            finalized.resources().forEach(resource -> resource.close());
            context.close();
        }
    }

    @Test
    void preparedSplatsUploadEveryCarrierWidthExactlyOnce() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        List<ScalarValue> scalars = List.of(
                ScalarValue.float64(Double.longBitsToDouble(0x7ff8_0000_0000_0042L)),
                ScalarValue.float32(Float.intBitsToFloat(0xffc0_0042)),
                ScalarValue.bfloat16Bits((short) 0x8001),
                ScalarValue.int32(Integer.MIN_VALUE),
                ScalarValue.int64(Long.MIN_VALUE + 7L),
                ScalarValue.bool(true));
        try {
            for (ScalarValue scalar : scalars) {
                int width = scalar.dataType().byteWidth();
                MetalPreparedSplatResource resource =
                        MetalPreparedSplatResource.create(context, width * 3L, scalar);
                var binding = resource.newRunBinding();
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment downloaded = arena.allocate(width * 3L, width);
                    MetalPreparedSplatResource.readableBuffer(binding)
                            .download(0L, downloaded, 0L, width * 3L);
                    for (long index = 0L; index < 3L; index++) {
                        switch (scalar.dataType()) {
                            case FLOAT64 -> assertEquals(
                                    Double.doubleToRawLongBits(scalar.float64Value()),
                                    downloaded.getAtIndex(
                                            java.lang.foreign.ValueLayout.JAVA_LONG, index));
                            case FLOAT32 -> assertEquals(
                                    Float.floatToRawIntBits(scalar.float32Value()),
                                    downloaded.getAtIndex(JAVA_INT, index));
                            case BFLOAT16 -> assertEquals(
                                    scalar.bfloat16Bits(),
                                    downloaded.getAtIndex(
                                            java.lang.foreign.ValueLayout.JAVA_SHORT, index));
                            case INT32 -> assertEquals(
                                    scalar.int32Value(),
                                    downloaded.getAtIndex(JAVA_INT, index));
                            case INT64 -> assertEquals(
                                    scalar.int64Value(),
                                    downloaded.getAtIndex(
                                            java.lang.foreign.ValueLayout.JAVA_LONG, index));
                            case BOOL -> assertEquals(
                                    (byte) 1,
                                    downloaded.getAtIndex(JAVA_BYTE, index));
                        }
                    }
                } finally {
                    binding.close();
                    resource.close();
                }
            }
            assertEquals(scalars.size(), api.bufferCreates.get());
            assertEquals(scalars.size(), api.uploads.get());
        } finally {
            context.close();
        }
    }

    @Test
    void positiveRankSplatsArePreparedOnceAndOutputsStayIsolatedAcrossRuns()
            throws Exception {
        RecordingNativeApi api = new RecordingNativeApi();
        SplatRoute route = prepareSplatRoute(api);
        MetalBufferRepresentation caller = null;
        try {
            assertEquals(route.splatBits().length, api.bufferCreates.get(),
                    "prepare owns exactly one buffer per source splat");
            assertEquals(route.splatBits().length, api.uploads.get(),
                    "prepare uploads each source splat exactly once");
            caller = route.context().createBuffer(8);
            uploadBits(caller, 0x3F800000, 0xC0000000);
            int coldBaseCreates = api.bufferCreates.get();
            int coldBaseUploads = api.uploads.get();

            var runner = new PreparedExecutionRunner();
            var first = runner.run(route.execution(), List.of(caller));
            List<MetalBufferRepresentation> firstOutputs = publishedBuffers(first);
            RecordingNativeApi.RunObservation firstRun = api.runs.getFirst();
            assertSplatRun(route, api, caller, firstRun, coldBaseCreates, coldBaseUploads);
            assertEquals(coldBaseCreates + route.targetCount(), api.bufferCreates.get());
            assertEquals(coldBaseUploads, api.uploads.get());
            Set<Long> persistentHandles = new java.util.HashSet<>(firstRun.splatHandles());
            persistentHandles.add(caller.executionHandle().carrier().address());
            first.close();
            assertEquals(persistentHandles, api.liveBufferHandles(),
                    "closing a result releases outputs but not prepared splats");

            int secondBaseCreates = api.bufferCreates.get();
            int secondBaseUploads = api.uploads.get();
            var second = runner.run(route.execution(), List.of(caller));
            List<MetalBufferRepresentation> secondOutputs = publishedBuffers(second);
            RecordingNativeApi.RunObservation secondRun = api.runs.get(1);
            assertSplatRun(route, api, caller, secondRun, secondBaseCreates, secondBaseUploads);
            assertNotSame(firstRun.inputAddresses(), secondRun.inputAddresses());
            assertEquals(firstRun.splatHandles(), secondRun.splatHandles());
            assertTrue(disjoint(firstRun.outputHandles(), secondRun.outputHandles()));
            for (int index = 0; index < firstOutputs.size(); index++) {
                assertNotSame(firstOutputs.get(index), secondOutputs.get(index));
            }
            second.close();
            assertEquals(route.splatBits().length + 2 * route.targetCount() + 1,
                    api.bufferCreates.get(),
                    "two runs add only outputs after prepared constants and caller input");
            assertEquals(route.splatBits().length + 1, api.uploads.get(),
                    "only prepared splats and the explicit caller upload occur");

            int concurrentRunBufferCount = route.targetCount();
            api.blockBufferCreates(concurrentRunBufferCount * 2);
            api.blockRuns(1);
            try (var executor = Executors.newFixedThreadPool(2)) {
                MetalBufferRepresentation sharedCaller = caller;
                var left = executor.submit(() -> runner.run(
                        route.execution(), List.of(sharedCaller)));
                var right = executor.submit(() -> runner.run(
                        route.execution(), List.of(sharedCaller)));
                assertTrue(api.bufferCreateEntered.await(5, TimeUnit.SECONDS));
                List<List<Long>> perThreadBuffers = api.concurrentCreatedHandles();
                assertEquals(2, perThreadBuffers.size());
                assertEquals(concurrentRunBufferCount, perThreadBuffers.get(0).size());
                assertEquals(concurrentRunBufferCount, perThreadBuffers.get(1).size());
                assertTrue(disjoint(perThreadBuffers.get(0), perThreadBuffers.get(1)));
                assertTrue(api.runEntered.await(5, TimeUnit.SECONDS));
                api.continueRuns.countDown();
                var leftResult = left.get(5, TimeUnit.SECONDS);
                var rightResult = right.get(5, TimeUnit.SECONDS);
                try {
                    List<RecordingNativeApi.RunObservation> concurrent =
                            api.runs.subList(api.runs.size() - 2, api.runs.size());
                    assertEquals(concurrent.get(0).splatHandles(),
                            concurrent.get(1).splatHandles());
                    assertTrue(disjoint(concurrent.get(0).outputHandles(),
                            concurrent.get(1).outputHandles()));
                    assertNotSame(concurrent.get(0).inputAddresses(),
                            concurrent.get(1).inputAddresses());
                    for (int index = 0; index < route.targetCount(); index++) {
                        assertNotSame(leftResult.publicationRepresentation(index),
                                rightResult.publicationRepresentation(index));
                    }
                } finally {
                    leftResult.close();
                    rightResult.close();
                }
            } finally {
                api.continueRuns.countDown();
            }
            assertEquals(persistentHandles, api.liveBufferHandles());
        } finally {
            close(route.execution());
            close(caller);
            route.context().close();
        }
        assertTrue(api.liveBufferHandles().isEmpty());
    }

    @Test
    void splatAllocationFailureDuringFinalizationRollsBackEarlierSplatThenExecutable() {
        RecordingNativeApi api = new RecordingNativeApi();
        RuntimeException primary = new RuntimeException("splat allocation");
        RuntimeException cleanup = new RuntimeException("earlier splat release");
        api.failBufferCreateCall = 2;
        api.bufferCreateFailure = primary;
        api.bufferReleaseFailures.add(cleanup);

        RuntimeException actual = assertThrows(
                RuntimeException.class, () -> prepareSplatRoute(api));
        assertSame(primary, actual);
        assertArrayEquals(new Throwable[] {cleanup}, actual.getSuppressed());
        assertEquals(0, api.runCalls.get());
        assertEquals(2, api.bufferCreates.get());
        assertEquals(1, api.bufferReleases.get());
        assertEquals(1, api.executableCreates.get());
        assertEquals(1, api.executableReleases.get());
        assertEquals(1, api.contextReleases.get());
        assertTrue(api.liveBufferHandles().isEmpty());
    }

    @Test
    void splatUploadFailureClosesCurrentThenPriorSplatBeforeExecutable() {
        RecordingNativeApi api = new RecordingNativeApi();
        RuntimeException primary = new RuntimeException("splat upload");
        RuntimeException currentCleanup = new RuntimeException("current splat release");
        RuntimeException earlierCleanup = new RuntimeException("earlier splat release");
        api.failUploadCall = 2;
        api.uploadFailure = primary;
        api.bufferReleaseFailures.add(currentCleanup);
        api.bufferReleaseFailures.add(earlierCleanup);

        RuntimeException actual = assertThrows(
                RuntimeException.class, () -> prepareSplatRoute(api));
        assertSame(primary, actual);
        assertArrayEquals(
                new Throwable[] {currentCleanup, earlierCleanup}, actual.getSuppressed());
        assertEquals(0, api.runCalls.get());
        assertEquals(2, api.bufferCreates.get());
        assertEquals(2, api.uploads.get());
        assertEquals(2, api.bufferReleases.get());
        assertEquals(1, api.executableCreates.get());
        assertEquals(1, api.executableReleases.get());
        assertEquals(1, api.contextReleases.get());
        assertTrue(api.liveBufferHandles().isEmpty());
    }

    @Test
    void realDeviceTypedMpsGraphInitializesLogicalSplatsPerRun() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        SplatRoute route = prepareSplatRoute(MetalDeviceContext.open(
                Path.of(configured).toAbsolutePath().normalize()));
        MetalBufferRepresentation caller = route.context().createBuffer(8);
        try (Arena arena = Arena.ofConfined()) {
            uploadBits(caller, 0x3F800000, 0xC0000000);
            assertEquals(MetalPreparedRoute.MPSGRAPH, route.plan().route());
            assertTrue(route.plan().feedSplats().getFirst().isEmpty());
            assertTrue(route.plan().feedSplats().subList(
                    1, route.plan().feedSplats().size()).stream().allMatch(Optional::isPresent));

            var runner = new PreparedExecutionRunner();
            var first = runner.run(route.execution(), List.of(caller));
            Object firstOutput = first.publicationRepresentation(0);
            try {
                assertLogicalSplatOutputs(first, arena);
            } finally {
                first.close();
            }
            var second = runner.run(route.execution(), List.of(caller));
            try {
                assertNotSame(firstOutput, second.publicationRepresentation(0));
                assertLogicalSplatOutputs(second, arena);
            } finally {
                second.close();
            }
        } finally {
            caller.close();
            close(route.execution());
            route.context().close();
        }
    }

    @Test
    void realDeviceMpsGraphStressReusesOneExecutableAndWritesSuppliedDestinations() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        MetalDeviceContext context = MetalDeviceContext.open(
                Path.of(configured).toAbsolutePath().normalize());
        MetalMpsGraphExecutableResource resource = null;
        MetalBufferRepresentation input0 = null;
        MetalBufferRepresentation input1 = null;
        MetalBufferRepresentation output0 = null;
        MetalBufferRepresentation output1 = null;
        MetalBufferRepresentation output2 = null;
        try {
            resource = context.createMpsGraphExecutable(analyze(fixture(), context).plan());
            input0 = context.createBuffer(24);
            input1 = context.createBuffer(16);
            output0 = context.createBuffer(24);
            output1 = context.createBuffer(16);
            output2 = context.createBuffer(24);
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment first = arena.allocate(24);
                MemorySegment second = arena.allocate(16);
                float[] a = {1.0f, -2.0f, 0.0f, -0.0f, Float.POSITIVE_INFINITY, Float.NaN};
                float[] b = {3.0f, -4.0f, Float.NEGATIVE_INFINITY, 0.0f};
                for (int index = 0; index < a.length; index++) first.setAtIndex(JAVA_FLOAT, index, a[index]);
                for (int index = 0; index < b.length; index++) second.setAtIndex(JAVA_FLOAT, index, b[index]);
                input0.upload(0, first, 0, 24);
                input1.upload(0, second, 0, 16);
                var workspace = addressWorkspace(
                        context, input0, input1, output0, output1, output2);
                resource.run(2, workspace.segment().asSlice(0, 16),
                        3, workspace.segment().asSlice(16, 24));
                assertNegated(output0, a, arena);
                assertNegated(output1, new float[] {
                        -3.0f, 4.0f, Float.POSITIVE_INFINITY, -0.0f
                }, arena);
                assertNegated(output2, new float[] {
                        -1.0f, 2.0f, -0.0f, 0.0f, Float.NEGATIVE_INFINITY, Float.NaN
                }, arena);
                for (int run = 0; run < MPSGRAPH_AUTORELEASE_STRESS_RUNS; run++) {
                    resource.run(2, workspace.segment().asSlice(0, 16),
                            3, workspace.segment().asSlice(16, 24));
                }
                assertNegated(output0, a, arena);
            }
        } finally {
            close(output2); close(output1); close(output0); close(input1); close(input0);
            close(resource); context.close();
        }
    }

    @Test
    void nativePreparedNegRoundTripAndReuse() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        MetalDeviceContext context = MetalDeviceContext.open(
                Path.of(configured).toAbsolutePath().normalize());
        var input0 = context.createBuffer(24);
        var input1 = context.createBuffer(16);
        io.github.pho001.synaptik.runtime.execution.PreparedExecution execution = null;
        io.github.pho001.synaptik.runtime.execution.PreparedExecution customCaller = null;
        io.github.pho001.synaptik.runtime.execution.PreparedExecution customSplat = null;
        try (Arena arena = Arena.ofConfined()) {
            Tensor firstInput = TensorFactory.create(descriptor(Shape.of(2, 3)));
            Tensor secondInput = TensorFactory.create(descriptor(Shape.of(4)));
            Tensor shared = firstInput.neg();
            Tensor chain = shared.neg();
            Tensor independent = secondInput.neg();
            var availability = new BackendAvailabilitySnapshot(
                    MetalCapabilityProvider.METAL_BACKEND_ID,
                    Map.of(new BackendDeviceId(MetalCapabilityProvider.METAL_BACKEND_ID, "default"),
                            DeviceClass.ACCELERATOR));
            var artifacts = GraphCompilationPort.compile(CompileMode.FORWARD_ONLY, io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, List.of(shared, chain, independent), Optional.empty(), GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(), PartitionScoringConfig.neutral(), List.of(new MetalCapabilityProvider()), List.of(availability));
            assertEquals(1, artifacts.partitions().size());

            var analyzed = new AtomicReference<MetalNegPreparationPlan>();
            var delegate = new MetalNegPartitionPreparer();
            var preparation = new PartitionPreparation<>(new MetalNegAnalysisInputs(context),
                    prepareContext -> {
                        var result = delegate.analyze(prepareContext);
                        analyzed.set(result.plan());
                        return result;
                    }, new MetalNegPartitionFinalizer(context));
            List<ValueId> publications = artifacts.publication().forwardBindings().stream()
                    .map(binding -> binding.valueId()).toList();
            execution = GraphPreparation.prepare(artifacts, List.of(preparation),
                    scheduleContext -> new MetalNegPreparedScheduleAssembler(
                            context, analyzed.get(), publications).assemble(scheduleContext));

            MemorySegment first = arena.allocate(24);
            MemorySegment second = arena.allocate(16);
            float[] a = {1.0f, -2.0f, 0.0f, -0.0f, Float.POSITIVE_INFINITY, Float.NaN};
            float[] b = {3.0f, -4.0f, Float.NEGATIVE_INFINITY, 0.0f};
            for (int index = 0; index < a.length; index++) first.setAtIndex(JAVA_FLOAT, index, a[index]);
            for (int index = 0; index < b.length; index++) second.setAtIndex(JAVA_FLOAT, index, b[index]);
            input0.upload(0, first, 0, 24);
            input1.upload(0, second, 0, 16);

            var runner = new PreparedExecutionRunner();
            customCaller = prepareSingleExecution(
                    context, Shape.of(2, 3), Optional.empty());
            var customFirst = runner.run(customCaller, List.of(input0));
            Object customFirstOutput = customFirst.publicationRepresentation(0);
            try {
                assertNegated((MetalBufferRepresentation) customFirstOutput,
                        new float[] {-1.0f, 2.0f, -0.0f, 0.0f,
                                Float.NEGATIVE_INFINITY, Float.NaN}, arena);
            } finally {
                customFirst.close();
            }
            var customSecond = runner.run(customCaller, List.of(input0));
            try {
                assertNotSame(customFirstOutput, customSecond.publicationRepresentation(0));
                assertNegated((MetalBufferRepresentation) customSecond.publicationRepresentation(0),
                        new float[] {-1.0f, 2.0f, -0.0f, 0.0f,
                                Float.NEGATIVE_INFINITY, Float.NaN}, arena);
            } finally {
                customSecond.close();
            }

            customSplat = prepareSingleExecution(context, Shape.of(4),
                    Optional.of(ScalarValue.float32(-3.5f)));
            var splatFirst = runner.run(customSplat, List.of());
            Object splatFirstOutput = splatFirst.publicationRepresentation(0);
            try {
                assertNegated((MetalBufferRepresentation) splatFirstOutput,
                        new float[] {3.5f, 3.5f, 3.5f, 3.5f}, arena);
            } finally {
                splatFirst.close();
            }
            var splatSecond = runner.run(customSplat, List.of());
            try {
                assertNotSame(splatFirstOutput, splatSecond.publicationRepresentation(0));
            } finally {
                splatSecond.close();
            }

            assertEquals(MetalPreparedRoute.MPSGRAPH, analyzed.get().route());
            var firstRun = runner.run(execution, List.of(input0, input1));
            Object firstOutput = firstRun.publicationRepresentation(0);
            try {
                assertNegated((MetalBufferRepresentation) firstRun.publicationRepresentation(0),
                        new float[] {-1.0f, 2.0f, -0.0f, 0.0f,
                                Float.NEGATIVE_INFINITY, Float.NaN}, arena);
                assertNegated((MetalBufferRepresentation) firstRun.publicationRepresentation(1),
                        a, arena);
                assertNegated((MetalBufferRepresentation) firstRun.publicationRepresentation(2),
                        new float[] {-3.0f, 4.0f, Float.POSITIVE_INFINITY, -0.0f}, arena);
            } finally {
                firstRun.close();
            }
            var secondRun = runner.run(execution, List.of(input0, input1));
            try {
                assertTrue(firstOutput != secondRun.publicationRepresentation(0));
                assertNegated((MetalBufferRepresentation) secondRun.publicationRepresentation(0),
                        new float[] {-1.0f, 2.0f, -0.0f, 0.0f,
                                Float.NEGATIVE_INFINITY, Float.NaN}, arena);
            } finally {
                secondRun.close();
            }
        } finally {
            close(customSplat); close(customCaller); close(execution);
            input1.close(); input0.close(); context.close();
        }
    }

    @Test
    void affineAnalysisAndFakeNativeFinalizationCarryEveryTypedMappingAndDenseGeometry() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        BackendPartitionFinalizationResult finalized = null;
        try {
            AffineFixture fixture = affineFixture();
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                    analyze(fixture, context);
            MetalNegPreparationPlan plan = analysis.plan();

            assertEquals(MetalPreparedRoute.CUSTOM_PROGRAM, plan.route());
            assertEquals(fixture.feeds(), plan.feedValueIds());
            assertEquals(fixture.targets(), plan.targetValueIds());
            assertArrayEquals(new int[] {0, 2, 4, 6, 8}, plan.feedValueIndices());
            assertArrayEquals(new int[] {1, 3, 5, 7, 9}, plan.targetValueIndices());
            assertArrayEquals(new long[] {24, 12, 24, 24, 24},
                    plan.feedRequiredBytes());
            assertArrayEquals(new long[] {24, 24, 24, 24, 24},
                    plan.targetRequiredBytes());
            assertEquals(10, plan.declarations().size());
            assertEquals(concat(fixture.feeds(), fixture.targets()),
                    plan.declarations().stream().map(
                            PreparationResourceRequirement.Buffer::valueId).toList());
            assertEquals(
                    List.of(24L, 12L, 24L, 24L, 24L, 24L, 24L, 24L, 24L, 24L),
                    plan.declarations().stream().map(
                            PreparationResourceRequirement.Buffer::byteSize).toList());
            assertEquals(120L, plan.addressWorkspace().orElseThrow().byteSize());

            List<MetalMpsGraphProgram.Node> nodes = plan.graphProgram().nodes();
            assertTypedNode(nodes.get(0), MetalMpsGraphProgram.NodeKind.RESHAPE,
                    0, 1, 2, MetalMpsGraphProgram.NO_AXIS, 3L, 2L);
            assertTypedNode(nodes.get(1), MetalMpsGraphProgram.NodeKind.EXPAND,
                    2, 3, 2, MetalMpsGraphProgram.NO_AXIS, 2L, 3L);
            assertTypedNode(nodes.get(2), MetalMpsGraphProgram.NodeKind.PERMUTE,
                    4, 5, 2, MetalMpsGraphProgram.NO_AXIS, 1L, 0L);
            assertTypedNode(nodes.get(3), MetalMpsGraphProgram.NodeKind.EXPAND_DIMS,
                    6, 7, 1, 1);
            assertTypedNode(nodes.get(4), MetalMpsGraphProgram.NodeKind.SQUEEZE,
                    8, 9, 1, 1);

            TensorDescriptor expanded = descriptorFor(fixture, fixture.targets().get(1));
            assertEquals(3L, expanded.layout().orElseThrow().referencedElementSpan());
            assertEquals(24L, plan.targetRequiredBytes()[1],
                    "dense affine declaration uses full logical geometry, not view span");

            FinalizationFixture assignment = finalization(analysis);
            finalized = new MetalNegPartitionFinalizer(context)
                    .finalizePartition(assignment.finalization());
            MetalNegPreparedExecutable executable =
                    (MetalNegPreparedExecutable) finalized.executable();
            for (ValueId target : fixture.targets()) {
                assertTrue(executable.denseAffinePublication(target).isPresent());
            }
            assertEquals(1, api.executableCreates.get());
            assertArrayEquals(
                    plan.graphProgram().encodedProgramImage(
                            plan.numericalProfile(),
                            plan.programValueDescriptors(),
                            plan.feedValueIndices(),
                            plan.targetValueIndices(),
                            plan.route()),
                    api.createdProgramImage);

            context.close();
            assertEquals(0, api.contextReleases.get(),
                    "the finalized affine executable retains the context lease");
            finalized.resources().getFirst().close();
            finalized.resources().getFirst().close();
            assertEquals(1, api.executableReleases.get());
            assertEquals(1, api.contextReleases.get());
            finalized = null;
        } finally {
            if (finalized != null) {
                finalized.resources().forEach(resource -> resource.close());
            }
            context.close();
        }
    }

    @Test
    void affineCompositionAuthenticatesLocalViewsAndKeepsContiguousTargetsCanonical() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        BackendPartitionFinalizationResult finalized = null;
        try {
            AffineFixture fixture = compositionFixture(true);
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis = analyze(fixture, context);
            MetalNegPreparationPlan plan = analysis.plan();
            assertEquals(List.of(fixture.feeds().getFirst()), plan.feedValueIds());
            assertEquals(fixture.targets(), plan.targetValueIds());
            assertEquals(List.of(
                    MetalMpsGraphProgram.ValueState.CANONICAL,
                    MetalMpsGraphProgram.ValueState.AFFINE_VIEW,
                    MetalMpsGraphProgram.ValueState.AFFINE_VIEW,
                    MetalMpsGraphProgram.ValueState.AFFINE_VIEW,
                    MetalMpsGraphProgram.ValueState.AFFINE_VIEW,
                    MetalMpsGraphProgram.ValueState.AFFINE_VIEW,
                    MetalMpsGraphProgram.ValueState.CANONICAL,
                    MetalMpsGraphProgram.ValueState.CANONICAL,
                    MetalMpsGraphProgram.ValueState.AFFINE_VIEW),
                    plan.valueStates());
            assertEquals(List.of(
                    MetalMpsGraphProgram.NodeKind.RESHAPE,
                    MetalMpsGraphProgram.NodeKind.EXPAND,
                    MetalMpsGraphProgram.NodeKind.PERMUTE,
                    MetalMpsGraphProgram.NodeKind.EXPAND_DIMS,
                    MetalMpsGraphProgram.NodeKind.SQUEEZE,
                    MetalMpsGraphProgram.NodeKind.CONTIGUOUS,
                    MetalMpsGraphProgram.NodeKind.NEG,
                    MetalMpsGraphProgram.NodeKind.RESHAPE),
                    plan.graphProgram().nodes().stream()
                            .map(MetalMpsGraphProgram.Node::kind).toList());
            assertArrayEquals(
                    new long[] {24L, 96L, 96L, 96L, 96L, 96L, 96L, 96L},
                    plan.targetRequiredBytes());

            FinalizationFixture assignment = finalization(analysis);
            finalized = new MetalNegPartitionFinalizer(context)
                    .finalizePartition(assignment.finalization());
            MetalNegPreparedExecutable executable =
                    (MetalNegPreparedExecutable) finalized.executable();
            for (int target = 0; target < fixture.targets().size(); target++) {
                boolean affineView = target <= 4 || target == 7;
                assertEquals(affineView,
                        executable.denseAffinePublication(fixture.targets().get(target)).isPresent());
            }
        } finally {
            if (finalized != null) {
                finalized.resources().forEach(resource -> resource.close());
            }
            context.close();
        }
    }

    @Test
    void affineCompositionRejectsForeignViewMalformedOrderAndViewToNegBeforeNativeWork() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        try {
            assertThrows(IllegalArgumentException.class,
                    () -> analyze(externalViewFeedFixture(), context));
            assertThrows(IllegalArgumentException.class,
                    () -> analyze(compositionFixture(false), context));

            AffineFixture ordered = compositionFixture(true);
            var reorderedNodes = new ArrayList<>(ordered.nodes());
            java.util.Collections.swap(reorderedNodes, 0, 1);
            AffineFixture reordered = new AffineFixture(
                    ordered.partition(),
                    List.copyOf(reorderedNodes),
                    ordered.values(),
                    ordered.requirements(),
                    ordered.feeds(),
                    ordered.targets());
            assertThrows(IllegalArgumentException.class, () -> analyze(reordered, context));
            assertEquals(0, api.executableCreates.get());
        } finally {
            context.close();
        }
    }

    @Test
    void everyMalformedAffineMappingRejectsBeforeFakeNativeAllocation() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        try {
            AffineFixture fixture = affineFixture();
            List<Operation> malformed = List.of(
                    new Operation(
                            ShapeTransformKind.RESHAPE,
                            new TargetShapeAttrs(Shape.of(2, 3))),
                    new Operation(
                            ShapeTransformKind.EXPAND,
                            new TargetShapeAttrs(Shape.of(1, 3))),
                    new Operation(
                            AxisTransformKind.PERMUTE,
                            new PermutationAttrs(List.of(0, 1))),
                    new Operation(
                            AxisTransformKind.EXPAND_DIMS,
                            new AxisTransformAttrs(0)),
                    new Operation(
                            AxisTransformKind.SQUEEZE,
                            new AxisTransformAttrs(0)));
            for (int index = 0; index < malformed.size(); index++) {
                AffineFixture rejected = withOperation(fixture, index, malformed.get(index));
                assertThrows(IllegalArgumentException.class,
                        () -> analyze(rejected, context));
                assertEquals(0, api.executableCreates.get(),
                        "mapping " + index + " must reject before native create");
                assertEquals(0, api.bufferCreates.get(),
                        "mapping " + index + " must reject before physical allocation");
            }
        } finally {
            context.close();
        }
    }

    @Test
    void affineFinalizerRollbackReleasesSharedExecutableAndPreservesCleanupFailure() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                analyze(affineFixture(), context);
        FinalizationFixture fixture = finalization(analysis);
        RuntimeException primary = new RuntimeException("affine recipe construction");
        api.releaseStatus = 7;
        var finalizer = new MetalNegPartitionFinalizer(context,
                (plan,
                        memoryPlan,
                        feeds,
                        feedRepresentations,
                        targets,
                        targetRepresentations,
                        internalPlanIndices,
                        internalRepresentationIndices,
                        resource,
                        splats,
                        feedBytes,
                        targetBytes,
                        internalBytes,
                        workspace) -> {
                    assertEquals(5, plan.graphProgram().nodes().size());
                    throw primary;
                });
        try {
            RuntimeException failure = assertThrows(RuntimeException.class,
                    () -> finalizer.finalizePartition(fixture.finalization()));
            assertSame(primary, failure);
            assertEquals(1, failure.getSuppressed().length);
            assertNativeFailure((MetalNativeApi.NativeFailure) failure.getSuppressed()[0],
                    MetalNativeApi.EXECUTABLE_RELEASE_OPERATION, 7);
            assertEquals(1, api.executableCreates.get());
            assertEquals(1, api.executableReleases.get());
            assertEquals(0, api.pipelineCreates.get());
        } finally {
            context.close();
        }
        assertEquals(1, api.contextReleases.get());
    }

    @Test
    void MetalAffinePublicationRequiresFinalizedRouteEvidenceAndUsesFullLogicalBytes() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        MetalBackendRuntime runtime = new MetalBackendRuntime(context);
        BackendPartitionFinalizationResult finalized = null;
        MetalBufferRepresentation authenticated = null;
        MetalBufferRepresentation unauthenticated = null;
        MetalBufferRepresentation canonical = null;
        try {
            Shape inputShape = Shape.of(1, 3);
            Shape outputShape = Shape.of(2, 3);
            TensorDescriptor inputDescriptor = descriptor(inputShape);
            TensorDescriptor outputDescriptor = new TensorDescriptor(
                    DataType.FLOAT32,
                    outputShape,
                    Optional.of(LayoutDescriptor.of(
                            outputShape, new long[] {0, 1}, 0L, true)),
                    false);
            ValueId feed = new ValueId(600);
            ValueId target = new ValueId(601);
            CompiledNode node = new CompiledNode(
                    new NodeId(600),
                    new Operation(
                            ShapeTransformKind.EXPAND,
                            new TargetShapeAttrs(outputShape)),
                    List.of(feed),
                    List.of(target));
            PlannedPartition partition = new PlannedPartition(
                    MetalCapabilityProvider.METAL_BACKEND_ID, List.of(node.id()));
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                    new MetalNegPartitionPreparer().analyze(new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, new PartitionDag(partition, List.of(node)), List.of(
                            new GraphValue(feed, inputDescriptor),
                            new GraphValue(target, outputDescriptor)), List.of(
                            requirement(
                                    feed,
                                    inputDescriptor,
                                    Optional.empty(),
                                    List.of(partition),
                                    false),
                            requirement(
                                    target,
                                    outputDescriptor,
                                    Optional.of(partition),
                                    List.of(),
                                    true)), Map.of(), new MetalNegAnalysisInputs(context)));
            FinalizationFixture assignment = finalization(analysis);
            finalized = new MetalNegPartitionFinalizer(context)
                    .finalizePartition(assignment.finalization());
            MetalNegPreparedExecutable executable =
                    (MetalNegPreparedExecutable) finalized.executable();

            authenticated = context.createBuffer(
                    24L, executable.denseAffinePublication(target).orElseThrow());
            unauthenticated = context.createBuffer(24L);
            canonical = context.createBuffer(12L);
            int[] denseBits = {
                    0x00000000, 0x80000000, 0x7fc12345,
                    0x00000000, 0x80000000, 0x7fc12345
            };
            uploadBits(authenticated, denseBits);
            uploadBits(unauthenticated, denseBits);
            uploadBits(canonical, 0x00000000, 0x80000000, 0x7fc12345);

            byte[] affineBytes = runtime.copyToCanonicalHostBytes(
                    authenticated, outputDescriptor, 24L);
            assertEquals(24, affineBytes.length);
            for (int index = 0; index < denseBits.length; index++) {
                assertEquals(denseBits[index],
                        java.nio.ByteBuffer.wrap(affineBytes)
                                .getInt(index * Integer.BYTES));
            }
            MetalBufferRepresentation rejectedView = unauthenticated;
            assertThrows(
                    IllegalArgumentException.class,
                    () -> runtime.copyToCanonicalHostBytes(
                            rejectedView, outputDescriptor, 24L));
            byte[] canonicalBytes = runtime.copyToCanonicalHostBytes(
                    canonical, inputDescriptor, 12L);
            assertEquals(12, canonicalBytes.length);
            assertEquals(0x7fc12345,
                    java.nio.ByteBuffer.wrap(canonicalBytes).getInt(8));
        } finally {
            if (canonical != null) {
                canonical.close();
            }
            if (unauthenticated != null) {
                unauthenticated.close();
            }
            if (authenticated != null) {
                authenticated.close();
            }
            if (finalized != null) {
                for (int index = finalized.resources().size() - 1; index >= 0; index--) {
                    finalized.resources().get(index).close();
                }
            }
            runtime.close();
        }
    }

    @Test
    void selectStoragePublicationGathersExactCanonicalBitsForAllCarriers() {
        RecordingNativeApi api = new RecordingNativeApi();
        MetalDeviceContext context = MetalDeviceContext.open(api);
        MetalBackendRuntime runtime = new MetalBackendRuntime(context);
        try {
            int ordinal = 0;
            for (DataType type : DataType.values()) {
                Shape inputShape = Shape.of(3, 4);
                Shape outputShape = Shape.of(4);
                LayoutDescriptor inputLayout =
                        LayoutDescriptor.of(inputShape, new long[] {6, 1}, 2L, true);
                LayoutDescriptor outputLayout =
                        LayoutDescriptor.of(outputShape, new long[] {1}, 8L, true);
                TensorDescriptor inputDescriptor = new TensorDescriptor(
                        type, inputShape, Optional.of(inputLayout), false);
                TensorDescriptor outputDescriptor = new TensorDescriptor(
                        type, outputShape, Optional.of(outputLayout), false);
                ValueId feed = new ValueId(700 + ordinal * 2L);
                ValueId target = new ValueId(701 + ordinal * 2L);
                CompiledNode node = new CompiledNode(
                        new NodeId(700 + ordinal),
                        new Operation(SelectKind.SELECT, new SelectAttrs(0, 1)),
                        List.of(feed),
                        List.of(target));
                PlannedPartition partition = new PlannedPartition(
                        MetalCapabilityProvider.METAL_BACKEND_ID, List.of(node.id()));
                BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                        new MetalNegPartitionPreparer().analyze(new PrepareContext<>(
                                NumericalProfile.STRICT_IEEE,
                                new PartitionDag(partition, List.of(node)),
                                List.of(
                                        new GraphValue(feed, inputDescriptor),
                                        new GraphValue(target, outputDescriptor)),
                                List.of(
                                        requirement(
                                                feed,
                                                inputDescriptor,
                                                Optional.empty(),
                                                List.of(partition),
                                                false),
                                        requirement(
                                                target,
                                                outputDescriptor,
                                                Optional.of(partition),
                                                List.of(),
                                                true)),
                                Map.of(),
                                new MetalNegAnalysisInputs(context)));
                FinalizationFixture assignment = finalization(analysis);
                BackendPartitionFinalizationResult finalized =
                        new MetalNegPartitionFinalizer(context)
                                .finalizePartition(assignment.finalization());
                MetalBufferRepresentation publication = null;
                try {
                    MetalNegPreparedExecutable executable =
                            (MetalNegPreparedExecutable) finalized.executable();
                    long spanBytes = Math.multiplyExact(
                            outputLayout.referencedElementSpan(), type.byteWidth());
                    publication = context.createBuffer(
                            spanBytes,
                            executable.denseAffinePublication(target).orElseThrow());
                    try (Arena arena = Arena.ofConfined()) {
                        MemorySegment physical =
                                arena.allocate(spanBytes, type.byteWidth());
                        physical.fill((byte) 0x5a);
                        for (int element = 0; element < 4; element++) {
                            for (int byteIndex = 0;
                                    byteIndex < type.byteWidth();
                                    byteIndex++) {
                                byte value = type == DataType.BOOL
                                        ? (byte) (element & 1)
                                        : (byte) (0x20 + element * type.byteWidth()
                                                + byteIndex);
                                physical.set(
                                        JAVA_BYTE,
                                        Math.multiplyExact(8L + element, type.byteWidth())
                                                + byteIndex,
                                        value);
                            }
                        }
                        publication.upload(0L, physical, 0L, spanBytes);
                        long canonicalBytes = Math.multiplyExact(4L, type.byteWidth());
                        byte[] canonical = runtime.copyToCanonicalHostBytes(
                                publication, outputDescriptor, canonicalBytes);
                        assertEquals(Math.toIntExact(canonicalBytes), canonical.length);
                        for (int element = 0; element < 4; element++) {
                            for (int byteIndex = 0;
                                    byteIndex < type.byteWidth();
                                    byteIndex++) {
                                int sourceByte = type == DataType.BOOL
                                                || java.nio.ByteOrder.nativeOrder()
                                                        == java.nio.ByteOrder.BIG_ENDIAN
                                        ? byteIndex : type.byteWidth() - 1 - byteIndex;
                                assertEquals(
                                        physical.get(
                                                JAVA_BYTE,
                                                Math.multiplyExact(
                                                        8L + element, type.byteWidth())
                                                        + sourceByte),
                                        canonical[element * type.byteWidth() + byteIndex],
                                        type + " element=" + element
                                                + " byte=" + byteIndex);
                            }
                        }
                    }
                } finally {
                    if (publication != null) publication.close();
                    for (int index = finalized.resources().size() - 1; index >= 0; index--) {
                        finalized.resources().get(index).close();
                    }
                }
                ordinal++;
            }
        } finally {
      runtime.close();
    }
    assertTrue(api.liveBufferHandles().isEmpty());
  }

  @Test
  void cropOfExpandedValuePublishesFromIndependentPhysicalOffsetAndSpan() {
    RecordingNativeApi api = new RecordingNativeApi();
    MetalDeviceContext context = MetalDeviceContext.open(api);
    MetalBackendRuntime runtime = new MetalBackendRuntime(context);
    BackendPartitionFinalizationResult finalized = null;
    MetalBufferRepresentation publication = null;
    try {
      Shape feedShape = Shape.of(1);
      Shape expandedShape = Shape.of(4);
      Shape targetShape = Shape.of(2);
      TensorDescriptor feedDescriptor =
          new TensorDescriptor(
              DataType.BOOL, feedShape, Optional.of(LayoutDescriptor.contiguous(feedShape)), false);
      TensorDescriptor expandedDescriptor =
          new TensorDescriptor(
              DataType.BOOL,
              expandedShape,
              Optional.of(LayoutDescriptor.of(expandedShape, new long[] {0L}, 0L, true)),
              false);
      TensorDescriptor targetDescriptor =
          new TensorDescriptor(
              DataType.BOOL,
              targetShape,
              Optional.of(LayoutDescriptor.of(targetShape, new long[] {0L}, 0L, true)),
              false);
      ValueId feed = new ValueId(760L);
      ValueId expanded = new ValueId(761L);
      ValueId target = new ValueId(762L);
      CompiledNode expand =
          new CompiledNode(
              new NodeId(760L),
              new Operation(ShapeTransformKind.EXPAND, new TargetShapeAttrs(expandedShape)),
              List.of(feed),
              List.of(expanded));
      CompiledNode crop =
          new CompiledNode(
              new NodeId(761L),
              new Operation(SliceKind.SLICE, new CropToShapeAttrs(targetShape, Shape.of(1))),
              List.of(expanded),
              List.of(target));
      PlannedPartition partition =
          new PlannedPartition(
              MetalCapabilityProvider.METAL_BACKEND_ID, List.of(expand.id(), crop.id()));
      BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
          new MetalNegPartitionPreparer()
              .analyze(
                  new PrepareContext<>(
                      NumericalProfile.STRICT_IEEE,
                      new PartitionDag(partition, List.of(expand, crop)),
                      List.of(
                          new GraphValue(feed, feedDescriptor),
                          new GraphValue(expanded, expandedDescriptor),
                          new GraphValue(target, targetDescriptor)),
                      List.of(
                          requirement(
                              feed, feedDescriptor, Optional.empty(), List.of(partition), false),
                          requirement(
                              expanded,
                              expandedDescriptor,
                              Optional.of(partition),
                              List.of(partition),
                              false),
                          requirement(
                              target, targetDescriptor, Optional.of(partition), List.of(), true)),
                      Map.of(),
                      new MetalNegAnalysisInputs(context)));
      int targetValue = analysis.plan().targetValueIndices()[0];
      LayoutDescriptor physicalTarget = analysis.plan().physicalLayouts().get(targetValue);
      assertEquals(LayoutDescriptor.of(targetShape, new long[] {1L}, 1L, true), physicalTarget);
      assertArrayEquals(new long[] {3L}, analysis.plan().targetRequiredBytes());
      FinalizationFixture assignment = finalization(analysis);
      finalized =
          new MetalNegPartitionFinalizer(context).finalizePartition(assignment.finalization());
      MetalNegPreparedExecutable executable = (MetalNegPreparedExecutable) finalized.executable();
      publication =
          context.createBuffer(3L, executable.denseAffinePublication(target).orElseThrow());
      try (Arena arena = Arena.ofConfined()) {
        MemorySegment physical = arena.allocate(3L, 1L);
        physical.setAtIndex(JAVA_BYTE, 0L, (byte) 0x5a);
        physical.setAtIndex(JAVA_BYTE, 1L, (byte) 1);
        physical.setAtIndex(JAVA_BYTE, 2L, (byte) 0);
        publication.upload(0L, physical, 0L, 3L);
        byte[] canonical = runtime.copyToCanonicalHostBytes(publication, targetDescriptor, 2L);
        assertArrayEquals(new byte[] {1, 0}, canonical);
      }
    } finally {
      if (publication != null) publication.close();
      if (finalized != null) {
        for (int index = finalized.resources().size() - 1; index >= 0; index--) {
          finalized.resources().get(index).close();
        }
      }
            runtime.close();
        }
        assertTrue(api.liveBufferHandles().isEmpty());
    }

    private static void assertNegated(
            MetalBufferRepresentation buffer, float[] expected, Arena arena) {
        MemorySegment bytes = arena.allocate((long) expected.length * Float.BYTES);
        buffer.download(0, bytes, 0, bytes.byteSize());
        for (int index = 0; index < expected.length; index++) {
            float actual = bytes.getAtIndex(JAVA_FLOAT, index);
            if (Float.isNaN(expected[index])) assertTrue(Float.isNaN(actual));
            else assertEquals(Float.floatToRawIntBits(expected[index]), Float.floatToRawIntBits(actual));
        }
    }

    private static void assertLogicalSplatOutputs(
            io.github.pho001.synaptik.runtime.run.RunResult result, Arena arena) {
        List<MetalBufferRepresentation> outputs = publishedBuffers(result);
        assertEquals(8, outputs.size());
        assertNegated(outputs.get(0), new float[] {-1.0f, 2.0f}, arena);
        assertNegated(outputs.get(1), new float[] {-0.0f, -0.0f}, arena);
        assertNegated(outputs.get(2), new float[] {0.0f, 0.0f}, arena);
        assertNegated(outputs.get(3), new float[] {
                Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY
        }, arena);
        assertNegated(outputs.get(4), new float[] {
                Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY
        }, arena);
        assertNegated(outputs.get(5), new float[] {Float.NaN, Float.NaN}, arena);
        assertNegated(outputs.get(6), new float[] {Float.NaN, Float.NaN}, arena);
        assertNegated(outputs.get(7), new float[] {Float.NaN, Float.NaN}, arena);
    }

    private static SplatRoute prepareSplatRoute(RecordingNativeApi api) {
        return prepareSplatRoute(MetalDeviceContext.open(api));
    }

    private static SplatRoute prepareSplatRoute(MetalDeviceContext context) {
        try {
            TensorDescriptor descriptor = descriptor(Shape.of(2));
            ValueId caller = new ValueId(100);
            int[] splatBits = {
                    0x00000000,
                    0x80000000,
                    0x7F800000,
                    0xFF800000,
                    0x7FC12345,
                    0x7FA12345
            };
            var splats = new ArrayList<ValueId>();
            for (int index = 0; index < splatBits.length; index++) {
                splats.add(new ValueId(101 + index));
            }
            var nodes = new ArrayList<CompiledNode>();
            var outputs = new ArrayList<ValueId>();
            Operation neg = new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE);
            ValueId callerOutput = new ValueId(200);
            outputs.add(callerOutput);
            nodes.add(new CompiledNode(new NodeId(1000), neg,
                    List.of(caller), List.of(callerOutput)));
            for (int index = 0; index < splats.size(); index++) {
                ValueId output = new ValueId(201 + index);
                outputs.add(output);
                nodes.add(new CompiledNode(new NodeId(1001 + index), neg,
                        List.of(splats.get(index)), List.of(output)));
                if (index == 4) {
                    ValueId repeatedOutput = new ValueId(300);
                    outputs.add(repeatedOutput);
                    nodes.add(new CompiledNode(new NodeId(1100), neg,
                            List.of(splats.get(index)), List.of(repeatedOutput)));
                }
            }
            PlannedPartition partition = new PlannedPartition(
                    MetalCapabilityProvider.METAL_BACKEND_ID,
                    nodes.stream().map(CompiledNode::id).toList());
            var values = new ArrayList<GraphValue>();
            values.add(new GraphValue(caller, descriptor));
            splats.forEach(value -> values.add(new GraphValue(value, descriptor)));
            outputs.forEach(value -> values.add(new GraphValue(value, descriptor)));
            var requirements = new ArrayList<LogicalMemoryRequirement>();
            requirements.add(requirement(caller, descriptor, Optional.empty(),
                    List.of(partition), false));
            for (ValueId splat : splats) {
                requirements.add(requirement(splat, descriptor, Optional.empty(),
                        List.of(partition), false));
            }
            for (ValueId output : outputs) {
                requirements.add(requirement(output, descriptor, Optional.of(partition),
                        List.of(), true));
            }
            var constants = new LinkedHashMap<ValueId, ScalarValue>();
            for (int index = 0; index < splats.size(); index++) {
                constants.put(splats.get(index), ScalarValue.float32(
                        Float.intBitsToFloat(splatBits[index])));
            }
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                    new MetalNegPartitionPreparer().analyze(new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, new PartitionDag(partition, nodes), values, requirements, constants, new MetalNegAnalysisInputs(context)));
            MetalNegPreparationPlan plan = analysis.plan();
            assertEquals(concat(List.of(caller), splats), plan.feedValueIds());
            int quietNanInput =
                    plan.graphProgram().nodes().get(5).firstInputIndex();
            assertEquals(
                    quietNanInput,
                    plan.graphProgram().nodes().get(6).firstInputIndex(),
                    "repeated consumption must retain one indexed feed");

            var bufferEntries = new ArrayList<PreparedMemoryPlan.BufferEntry>();
            var finalAssignments = new ArrayList<PreparationResourceAssignment>();
            var preparedAssignments = new ArrayList<PreparedBufferAssignment>();
            int bufferIndex = 0;
            for (var declaration : plan.declarations()) {
                BufferSlot slot = new BufferSlot(500 + bufferIndex);
                bufferEntries.add(new PreparedMemoryPlan.BufferEntry(
                        slot, declaration.byteSize(), declaration.byteAlignment()));
                finalAssignments.add(new PreparationResourceAssignment.Buffer(declaration, slot, bufferIndex, 0));
                preparedAssignments.add(new PreparedBufferAssignment(declaration.valueId(), slot, bufferIndex, java.util.List.of(io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider.METAL_BACKEND_ID)));
                bufferIndex++;
            }
            WorkspaceSlot workspaceSlot = new WorkspaceSlot(900);
            var workspaceRequirement = plan.addressWorkspace().orElseThrow();
            var memoryPlan = new PreparedMemoryPlan(bufferEntries, List.of(
                    new PreparedMemoryPlan.WorkspaceEntry(workspaceSlot,
                            workspaceRequirement.byteSize(),
                            workspaceRequirement.byteAlignment())));
            finalAssignments.add(new PreparationResourceAssignment.Workspace(
                    workspaceRequirement, workspaceSlot, 0));
            BackendPartitionFinalizationResult finalized =
                    new MetalNegPartitionFinalizer(context).finalizePartition(
                            new BackendPartitionFinalization<>(
                                    analysis, memoryPlan, finalAssignments));
            PreparedPartition preparedPartition = new PreparedPartition(
                    partition, finalized.executable());
            var schedule = new MetalNegPreparedScheduleAssembler(
                    context, plan, outputs).assembleRoute(
                            memoryPlan, preparedPartition, preparedAssignments);
            PreparedExecution execution = new PreparedExecution(
                    memoryPlan, schedule, finalized.resources());
            return new SplatRoute(context, execution, plan, splatBits, outputs.size());
        } catch (RuntimeException | Error failure) {
            context.close();
            throw failure;
        }
    }

    private static void assertSplatRun(
            SplatRoute route,
            RecordingNativeApi api,
            MetalBufferRepresentation caller,
            RecordingNativeApi.RunObservation run,
            int baseCreates,
            int baseUploads) {
        assertEquals(1 + route.splatBits().length, run.inputHandles().size());
        assertEquals(caller.executionHandle().carrier().address(), run.inputHandles().getFirst());
        assertEquals(route.targetCount(), run.outputHandles().size());
        assertEquals(baseCreates + route.targetCount(), run.bufferCreateCount());
        assertEquals(baseUploads, run.uploadCount());
        assertEquals(run.bufferCreateCount(), api.bufferCreates.get(),
                "hot execution must not allocate a Metal buffer");
        assertEquals(run.uploadCount(), api.uploads.get(),
                "hot execution must not upload a splat");
        for (int splat = 0; splat < route.splatBits().length; splat++) {
            assertArrayEquals(new int[] {route.splatBits()[splat], route.splatBits()[splat]},
                    run.inputRawBits().get(splat + 1));
        }
    }

    private static List<MetalBufferRepresentation> publishedBuffers(
            io.github.pho001.synaptik.runtime.run.RunResult result) {
        var buffers = new ArrayList<MetalBufferRepresentation>();
        for (int index = 0; index < result.resultCount(); index++) {
            buffers.add((MetalBufferRepresentation) result.publicationRepresentation(index));
        }
        return buffers;
    }

    private static void uploadBits(MetalBufferRepresentation buffer, int... bits) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = arena.allocate((long) bits.length * Integer.BYTES);
            for (int index = 0; index < bits.length; index++) {
                source.setAtIndex(JAVA_INT, index, bits[index]);
            }
            buffer.upload(0, source, 0, source.byteSize());
        }
    }

    private static boolean disjoint(List<Long> first, List<Long> second) {
        return first.stream().noneMatch(second::contains);
    }

    private static List<ValueId> concat(List<ValueId> first, List<ValueId> second) {
        var values = new ArrayList<ValueId>(first);
        values.addAll(second);
        return values;
    }

    private static MetalNegPreparedExecutable.AddressWorkspace addressWorkspace(
            MetalDeviceContext context, MetalBufferRepresentation... buffers) {
        var workspace = new MetalNegPreparedExecutable.AddressWorkspace(context, buffers.length);
        for (int index = 0; index < buffers.length; index++) {
            workspace.set(index, buffers[index].executionHandle());
        }
        return workspace;
    }

    private static MetalNegPreparationPlan analyze(Fixture fixture) {
        MetalDeviceContext marker = MetalDeviceContext.open(new RecordingNativeApi());
        try {
            return analyze(fixture, marker).plan();
        } finally {
            marker.close();
        }
    }

    private static BackendPartitionAnalysis<MetalNegPreparationPlan> analyze(
            Fixture fixture, MetalDeviceContext context) {
        return analyze(
                fixture,
                context,
                NumericalProfile.STRICT_IEEE);
    }

    private static BackendPartitionAnalysis<MetalNegPreparationPlan> analyze(
            Fixture fixture,
            MetalDeviceContext context,
            NumericalProfile numericalProfile) {
        return new MetalNegPartitionPreparer().analyze(new PrepareContext<>(
                numericalProfile,
                new PartitionDag(fixture.partition, fixture.nodes),
                fixture.values,
                fixture.requirements,
                Map.of(),
                new MetalNegAnalysisInputs(context)));
    }
    private static BackendPartitionAnalysis<MetalNegPreparationPlan> analyze(
            AffineFixture fixture, MetalDeviceContext context) {
        return new MetalNegPartitionPreparer().analyze(new PrepareContext<>(io.github.pho001.synaptik.config.compile.NumericalProfile.STRICT_IEEE, new PartitionDag(fixture.partition(), fixture.nodes()), fixture.values(), fixture.requirements(), Map.of(), new MetalNegAnalysisInputs(context)));
    }

    private static void assertTypedNode(
            MetalMpsGraphProgram.Node node,
            MetalMpsGraphProgram.NodeKind kind,
            int input,
            int output,
            int attributeCount,
            int axis,
            long... attributeValues) {
        assertEquals(kind, node.kind());
        assertEquals(input, node.firstInputIndex());
        assertEquals(MetalMpsGraphProgram.NO_SECOND_INPUT, node.secondInputIndex());
        assertEquals(output, node.outputIndex());
        assertEquals(attributeCount, node.attributeCount());
        assertEquals(axis, node.axis());
        assertArrayEquals(attributeValues, node.attributeValues());
    }

    private static AffineFixture affineFixture() {
        List<ValueId> feeds = List.of(
                new ValueId(20_000),
                new ValueId(20_010),
                new ValueId(20_020),
                new ValueId(20_030),
                new ValueId(20_040));
        List<ValueId> targets = List.of(
                new ValueId(20_001),
                new ValueId(20_011),
                new ValueId(20_021),
                new ValueId(20_031),
                new ValueId(20_041));
        List<TensorDescriptor> inputs = List.of(
                descriptor(Shape.of(2, 3)),
                descriptor(Shape.of(1, 3)),
                descriptor(Shape.of(2, 3)),
                descriptor(Shape.of(2, 3)),
                descriptor(Shape.of(2, 1, 3)));
        List<TensorDescriptor> outputs = List.of(
                viewDescriptor(Shape.of(3, 2), 2, 1),
                viewDescriptor(Shape.of(2, 3), 0, 1),
                viewDescriptor(Shape.of(3, 2), 1, 3),
                viewDescriptor(Shape.of(2, 1, 3), 3, 3, 1),
                viewDescriptor(Shape.of(2, 3), 3, 1));
        List<Operation> operations = List.of(
                new Operation(
                        ShapeTransformKind.RESHAPE,
                        new TargetShapeAttrs(Shape.of(3, 2))),
                new Operation(
                        ShapeTransformKind.EXPAND,
                        new TargetShapeAttrs(Shape.of(2, 3))),
                new Operation(
                        AxisTransformKind.PERMUTE,
                        new PermutationAttrs(List.of(1, 0))),
                new Operation(
                        AxisTransformKind.EXPAND_DIMS,
                        new AxisTransformAttrs(1)),
                new Operation(
                        AxisTransformKind.SQUEEZE,
                        new AxisTransformAttrs(1)));
        var nodes = new ArrayList<CompiledNode>();
        var values = new ArrayList<GraphValue>();
        for (int index = 0; index < operations.size(); index++) {
            nodes.add(new CompiledNode(
                    new NodeId(20_000 + index),
                    operations.get(index),
                    List.of(feeds.get(index)),
                    List.of(targets.get(index))));
            values.add(new GraphValue(feeds.get(index), inputs.get(index)));
            values.add(new GraphValue(targets.get(index), outputs.get(index)));
        }
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var requirements = new ArrayList<LogicalMemoryRequirement>();
        for (int index = 0; index < operations.size(); index++) {
            requirements.add(requirement(
                    feeds.get(index),
                    inputs.get(index),
                    Optional.empty(),
                    List.of(partition),
                    false));
            requirements.add(requirement(
                    targets.get(index),
                    outputs.get(index),
                    Optional.of(partition),
                    List.of(),
                    true));
        }
        return new AffineFixture(
                partition,
                List.copyOf(nodes),
                List.copyOf(values),
                List.copyOf(requirements),
                feeds,
                targets);
    }

    private static AffineFixture compositionFixture(boolean contiguousBarrier) {
        var values = new ArrayList<GraphValue>();
        var nodes = new ArrayList<CompiledNode>();
        var feeds = new ArrayList<ValueId>();
        var targets = new ArrayList<ValueId>();
        ValueId v0 = new ValueId(21_000);
        ValueId v1 = new ValueId(21_001);
        ValueId v2 = new ValueId(21_002);
        ValueId v3 = new ValueId(21_003);
        ValueId v4 = new ValueId(21_004);
        ValueId v5 = new ValueId(21_005);
        ValueId v6 = new ValueId(21_006);
        ValueId v7 = new ValueId(21_007);
        ValueId v8 = new ValueId(21_008);
        feeds.add(v0);
        values.add(new GraphValue(v0, descriptor(Shape.of(6))));
        values.add(new GraphValue(v1,
                viewDescriptor(Shape.of(2, 1, 3), 3, 3, 1)));
        values.add(new GraphValue(v2,
                viewDescriptor(Shape.of(2, 4, 3), 3, 0, 1)));
        values.add(new GraphValue(v3,
                viewDescriptor(Shape.of(4, 2, 3), 0, 3, 1)));
        values.add(new GraphValue(v4,
                viewDescriptor(Shape.of(4, 2, 1, 3), 0, 3, 3, 1)));
        values.add(new GraphValue(v5,
                viewDescriptor(Shape.of(4, 2, 3), 0, 3, 1)));
        if (contiguousBarrier) {
            values.add(new GraphValue(v6, descriptor(Shape.of(4, 2, 3))));
        }
        values.add(new GraphValue(v7, descriptor(Shape.of(4, 2, 3))));
        values.add(new GraphValue(v8,
                viewDescriptor(Shape.of(4, 6), 6, 1)));

        nodes.add(new CompiledNode(
                new NodeId(21_000),
                new Operation(
                        ShapeTransformKind.RESHAPE,
                        new TargetShapeAttrs(Shape.of(2, 1, 3))),
                List.of(v0),
                List.of(v1)));
        nodes.add(new CompiledNode(
                new NodeId(21_001),
                new Operation(
                        ShapeTransformKind.EXPAND,
                        new TargetShapeAttrs(Shape.of(2, 4, 3))),
                List.of(v1),
                List.of(v2)));
        nodes.add(new CompiledNode(
                new NodeId(21_002),
                new Operation(
                        AxisTransformKind.PERMUTE,
                        new PermutationAttrs(List.of(1, 0, 2))),
                List.of(v2),
                List.of(v3)));
        nodes.add(new CompiledNode(
                new NodeId(21_003),
                new Operation(
                        AxisTransformKind.EXPAND_DIMS,
                        new AxisTransformAttrs(2)),
                List.of(v3),
                List.of(v4)));
        nodes.add(new CompiledNode(
                new NodeId(21_004),
                new Operation(
                        AxisTransformKind.SQUEEZE,
                        new AxisTransformAttrs(2)),
                List.of(v4),
                List.of(v5)));
        ValueId negInput;
        if (contiguousBarrier) {
            nodes.add(new CompiledNode(
                    new NodeId(21_005),
                    new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE),
                    List.of(v5),
                    List.of(v6)));
            targets.add(v6);
            negInput = v6;
        } else {
            negInput = v5;
        }
        nodes.add(new CompiledNode(
                new NodeId(21_006),
                new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                List.of(negInput),
                List.of(v7)));
        nodes.add(new CompiledNode(
                new NodeId(21_007),
                new Operation(
                        ShapeTransformKind.RESHAPE,
                        new TargetShapeAttrs(Shape.of(4, 6))),
                List.of(v7),
                List.of(v8)));
        targets.addAll(0, List.of(v1, v2, v3, v4, v5));
        targets.add(v7);
        targets.add(v8);

        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        var requirements = new ArrayList<LogicalMemoryRequirement>();
        for (GraphValue value : values) {
            ValueId valueId = value.id();
            boolean feed = valueId.equals(v0);
            boolean finalValue = valueId.equals(v8);
            requirements.add(requirement(
                    valueId,
                    value.descriptor(),
                    feed ? Optional.empty() : Optional.of(partition),
                    finalValue ? List.of() : List.of(partition),
                    !feed));
        }
        return new AffineFixture(
                partition,
                List.copyOf(nodes),
                List.copyOf(values),
                List.copyOf(requirements),
                List.copyOf(feeds),
                List.copyOf(targets));
    }

    private static AffineFixture externalViewFeedFixture() {
        ValueId feed = new ValueId(22_000);
        ValueId target = new ValueId(22_001);
        TensorDescriptor input = viewDescriptor(Shape.of(2, 4, 3), 3, 0, 1);
        TensorDescriptor output = viewDescriptor(Shape.of(4, 2, 3), 0, 3, 1);
        CompiledNode node = new CompiledNode(
                new NodeId(22_000),
                new Operation(
                        AxisTransformKind.PERMUTE,
                        new PermutationAttrs(List.of(1, 0, 2))),
                List.of(feed),
                List.of(target));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID, List.of(node.id()));
        return new AffineFixture(
                partition,
                List.of(node),
                List.of(new GraphValue(feed, input), new GraphValue(target, output)),
                List.of(
                        requirement(feed, input, Optional.empty(), List.of(partition), false),
                        requirement(target, output, Optional.of(partition), List.of(), true)),
                List.of(feed),
                List.of(target));
    }

    private static AffineFixture withOperation(
            AffineFixture fixture, int nodeIndex, Operation operation) {
        var nodes = new ArrayList<>(fixture.nodes());
        CompiledNode existing = nodes.get(nodeIndex);
        nodes.set(nodeIndex, new CompiledNode(
                existing.id(), operation, existing.inputs(), existing.outputs()));
        return new AffineFixture(
                fixture.partition(),
                List.copyOf(nodes),
                fixture.values(),
                fixture.requirements(),
                fixture.feeds(),
                fixture.targets());
    }

    private static TensorDescriptor descriptorFor(
            AffineFixture fixture, ValueId valueId) {
        return fixture.values().stream()
                .filter(value -> value.id().equals(valueId))
                .findFirst()
                .orElseThrow()
                .descriptor();
    }

    private static TensorDescriptor viewDescriptor(Shape shape, long... strides) {
        return new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.of(shape, strides, 0L, true)),
                false);
    }

    private static Fixture binaryFixture() {
        TensorDescriptor matrix = descriptor(Shape.of(2, 3));
        TensorDescriptor row = descriptor(Shape.of(3));
        ValueId v0 = new ValueId(30_000);
        ValueId v1 = new ValueId(30_001);
        ValueId v2 = new ValueId(30_002);
        ValueId v3 = new ValueId(30_003);
        ValueId v4 = new ValueId(30_004);
        ValueId v5 = new ValueId(30_005);
        List<CompiledNode> nodes = List.of(
                new CompiledNode(
                        new NodeId(30_000),
                        new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                        List.of(v0, v1),
                        List.of(v2)),
                new CompiledNode(
                        new NodeId(30_001),
                        new Operation(BinaryArithmeticKind.SUB, NoOperationAttrs.INSTANCE),
                        List.of(v1, v0),
                        List.of(v3)),
                new CompiledNode(
                        new NodeId(30_002),
                        new Operation(BinaryArithmeticKind.MUL, NoOperationAttrs.INSTANCE),
                        List.of(v0, v0),
                        List.of(v4)),
                new CompiledNode(
                        new NodeId(30_003),
                        new Operation(BinaryArithmeticKind.DIV, NoOperationAttrs.INSTANCE),
                        List.of(v2, v1),
                        List.of(v5)));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        List<GraphValue> values = List.of(
                new GraphValue(v0, matrix),
                new GraphValue(v1, row),
                new GraphValue(v2, matrix),
                new GraphValue(v3, matrix),
                new GraphValue(v4, matrix),
                new GraphValue(v5, matrix));
        List<LogicalMemoryRequirement> requirements = List.of(
                requirement(v0, matrix, Optional.empty(), List.of(partition), false),
                requirement(v1, row, Optional.empty(), List.of(partition), false),
                requirement(v2, matrix, Optional.of(partition), List.of(partition), true),
                requirement(v3, matrix, Optional.of(partition), List.of(), true),
                requirement(v4, matrix, Optional.of(partition), List.of(), true),
                requirement(v5, matrix, Optional.of(partition), List.of(), true));
        return new Fixture(
                partition, nodes, values, requirements, v0, v1, v2, v3, v4, v5);
    }

    private static Fixture matmulFixture() {
        TensorDescriptor leftSource = descriptor(Shape.of(3, 2));
        TensorDescriptor rightSource = descriptor(Shape.of(4, 3));
        TensorDescriptor left = viewDescriptor(Shape.of(2, 3), 1, 2);
        TensorDescriptor right = viewDescriptor(Shape.of(3, 4), 1, 3);
        TensorDescriptor output = descriptor(Shape.of(2, 4));
        ValueId v0 = new ValueId(32_000);
        ValueId v1 = new ValueId(32_001);
        ValueId v2 = new ValueId(32_002);
        ValueId v3 = new ValueId(32_003);
        ValueId v4 = new ValueId(32_004);
        List<CompiledNode> nodes = List.of(
                new CompiledNode(
                        new NodeId(32_000),
                        new Operation(
                                AxisTransformKind.PERMUTE,
                                new PermutationAttrs(List.of(1, 0))),
                        List.of(v0),
                        List.of(v2)),
                new CompiledNode(
                        new NodeId(32_001),
                        new Operation(
                                AxisTransformKind.PERMUTE,
                                new PermutationAttrs(List.of(1, 0))),
                        List.of(v1),
                        List.of(v3)),
                new CompiledNode(
                        new NodeId(32_002),
                        new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                        List.of(v2, v3),
                        List.of(v4)));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        List<GraphValue> values = List.of(
                new GraphValue(v0, leftSource),
                new GraphValue(v1, rightSource),
                new GraphValue(v2, left),
                new GraphValue(v3, right),
                new GraphValue(v4, output));
        List<LogicalMemoryRequirement> requirements = List.of(
                requirement(v0, leftSource, Optional.empty(), List.of(partition), false),
                requirement(v1, rightSource, Optional.empty(), List.of(partition), false),
                requirement(v2, left, Optional.of(partition), List.of(partition), false),
                requirement(v3, right, Optional.of(partition), List.of(partition), false),
                requirement(v4, output, Optional.of(partition), List.of(), true));
        return new Fixture(
                partition, nodes, values, requirements, v0, v1, v2, v3, v4, v4);
    }
    private static Fixture batchedMatmulFixture() {
        TensorDescriptor left = descriptor(Shape.of(2, 2, 3));
        TensorDescriptor right = descriptor(Shape.of(1, 3, 4));
        TensorDescriptor output = descriptor(Shape.of(2, 2, 4));
        ValueId v0 = new ValueId(32_100);
        ValueId v1 = new ValueId(32_101);
        ValueId v2 = new ValueId(32_102);
        CompiledNode node = new CompiledNode(
                new NodeId(32_100),
                new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(v0, v1),
                List.of(v2));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID, List.of(node.id()));
        return new Fixture(
                partition,
                List.of(node),
                List.of(
                        new GraphValue(v0, left),
                        new GraphValue(v1, right),
                        new GraphValue(v2, output)),
                List.of(
                        requirement(v0, left, Optional.empty(), List.of(partition), false),
                        requirement(v1, right, Optional.empty(), List.of(partition), false),
                        requirement(v2, output, Optional.of(partition), List.of(), true)),
                v0, v1, v2, v2, v2, v2);
    }


    private static Fixture reductionFixture() {
        TensorDescriptor input = descriptor(Shape.of(2, 3, 4));
        TensorDescriptor matrix = descriptor(Shape.of(2, 4));
        TensorDescriptor scalar = descriptor(Shape.scalar());
        TensorDescriptor row = descriptor(Shape.of(1, 4));
        ValueId v0 = new ValueId(31_000);
        ValueId v1 = new ValueId(31_001);
        ValueId v2 = new ValueId(31_002);
        ValueId v3 = new ValueId(31_003);
        ValueId v4 = new ValueId(31_004);
        ValueId v5 = new ValueId(31_005);
        List<CompiledNode> nodes = List.of(
                new CompiledNode(
                        new NodeId(31_000),
                        new Operation(
                                AggregateReductionKind.MEAN,
                                new AxisReductionAttrs(1, false)),
                        List.of(v0),
                        List.of(v1)),
                new CompiledNode(
                        new NodeId(31_001),
                        new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE),
                        List.of(v1),
                        List.of(v2)),
                new CompiledNode(
                        new NodeId(31_002),
                        new Operation(AggregateReductionKind.SUM, NoOperationAttrs.INSTANCE),
                        List.of(v0),
                        List.of(v3)),
                new CompiledNode(
                        new NodeId(31_003),
                        new Operation(
                                AggregateReductionKind.SUM,
                                new SumToShapeAttrs(Shape.of(1, 4))),
                        List.of(v0),
                        List.of(v4)),
                new CompiledNode(
                        new NodeId(31_004),
                        new Operation(
                                AggregateReductionKind.SUM,
                                new MultiAxisReductionAttrs(List.of(), false)),
                        List.of(v0),
                        List.of(v5)));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        List<GraphValue> values = List.of(
                new GraphValue(v0, input),
                new GraphValue(v1, matrix),
                new GraphValue(v2, matrix),
                new GraphValue(v3, scalar),
                new GraphValue(v4, row),
                new GraphValue(v5, input));
        List<LogicalMemoryRequirement> requirements = List.of(
                requirement(v0, input, Optional.empty(), List.of(partition), false),
                requirement(v1, matrix, Optional.of(partition), List.of(partition), false),
                requirement(v2, matrix, Optional.of(partition), List.of(), true),
                requirement(v3, scalar, Optional.of(partition), List.of(), true),
                requirement(v4, row, Optional.of(partition), List.of(), true),
                requirement(v5, input, Optional.of(partition), List.of(), true));
        return new Fixture(
                partition, nodes, values, requirements, v0, v1, v2, v3, v4, v5);
    }

    private static Fixture mseFixture() {
        Shape shape = Shape.of(2, 3);
        TensorDescriptor prediction = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true);
        TensorDescriptor target = descriptor(shape);
        TensorDescriptor unreduced = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                true);
        Shape scalarShape = Shape.scalar();
        TensorDescriptor scalar = new TensorDescriptor(
                DataType.FLOAT32,
                scalarShape,
                Optional.of(LayoutDescriptor.contiguous(scalarShape)),
                true);
        ValueId v0 = new ValueId(31_100);
        ValueId v1 = new ValueId(31_101);
        ValueId v2 = new ValueId(31_102);
        ValueId v3 = new ValueId(31_103);
        ValueId v4 = new ValueId(31_104);
        List<LossReduction> reductions =
                List.of(LossReduction.NONE, LossReduction.SUM, LossReduction.MEAN);
        List<ValueId> outputs = List.of(v2, v3, v4);
        List<CompiledNode> nodes = new ArrayList<>();
        for (int index = 0; index < reductions.size(); index++) {
            nodes.add(new CompiledNode(
                    new NodeId(31_100 + index),
                    new Operation(
                            LossKind.MEAN_SQUARED_ERROR,
                            new MeanSquaredErrorAttrs(reductions.get(index))),
                    List.of(v0, v1),
                    List.of(outputs.get(index))));
        }
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        List<GraphValue> values = List.of(
                new GraphValue(v0, prediction),
                new GraphValue(v1, target),
                new GraphValue(v2, unreduced),
                new GraphValue(v3, scalar),
                new GraphValue(v4, scalar));
        List<LogicalMemoryRequirement> requirements = List.of(
                requirement(v0, prediction, Optional.empty(), List.of(partition), false),
                requirement(v1, target, Optional.empty(), List.of(partition), false),
                requirement(v2, unreduced, Optional.of(partition), List.of(), true),
                requirement(v3, scalar, Optional.of(partition), List.of(), true),
                requirement(v4, scalar, Optional.of(partition), List.of(), true));
        return new Fixture(
                partition, List.copyOf(nodes), values, requirements, v0, v1, v2, v3, v4, v4);
    }

    private static IndexingRoute indexingRoute(
            MetalDeviceContext context, Map<ValueId, ScalarValue> constants) {
        return indexingRoute(context, constants, null);
    }

    private static IndexingRoute indexingRoute(
            MetalDeviceContext context,
            Map<ValueId, ScalarValue> constants,
            MetalTraceProducer traceProducer) {
        ValueId data = new ValueId(40_000);
        ValueId gatherIndices = new ValueId(40_001);
        ValueId gatherOutput = new ValueId(40_002);
        ValueId oneHotIndices = new ValueId(40_003);
        ValueId oneHotOutput = new ValueId(40_004);
        TensorDescriptor dataDescriptor =
                typedDescriptor(DataType.FLOAT32, Shape.of(2, 3));
        TensorDescriptor gatherIndicesDescriptor =
                typedDescriptor(DataType.INT32, Shape.of(2));
        TensorDescriptor gatherOutputDescriptor =
                typedDescriptor(DataType.FLOAT32, Shape.of(2, 2));
        TensorDescriptor oneHotIndicesDescriptor =
                typedDescriptor(DataType.INT32, Shape.of(2));
        TensorDescriptor oneHotOutputDescriptor =
                typedDescriptor(DataType.BOOL, Shape.of(2, 4));
        List<CompiledNode> nodes = List.of(
                new CompiledNode(
                        new NodeId(40_000),
                        new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)),
                        List.of(data, gatherIndices),
                        List.of(gatherOutput)),
                new CompiledNode(
                        new NodeId(40_001),
                        new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(4)),
                        List.of(oneHotIndices),
                        List.of(oneHotOutput)));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        List<GraphValue> values = List.of(
                new GraphValue(data, dataDescriptor),
                new GraphValue(gatherIndices, gatherIndicesDescriptor),
                new GraphValue(gatherOutput, gatherOutputDescriptor),
                new GraphValue(oneHotIndices, oneHotIndicesDescriptor),
                new GraphValue(oneHotOutput, oneHotOutputDescriptor));
        List<LogicalMemoryRequirement> requirements = List.of(
                requirement(data, dataDescriptor, Optional.empty(), List.of(partition), false),
                requirement(gatherIndices, gatherIndicesDescriptor,
                        Optional.empty(), List.of(partition), false),
                requirement(gatherOutput, gatherOutputDescriptor,
                        Optional.of(partition), List.of(), true),
                requirement(oneHotIndices, oneHotIndicesDescriptor,
                        Optional.empty(), List.of(partition), false),
                requirement(oneHotOutput, oneHotOutputDescriptor,
                        Optional.of(partition), List.of(), true));
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                new MetalNegPartitionPreparer().analyze(new PrepareContext<>(
                        NumericalProfile.STRICT_IEEE,
                        new PartitionDag(partition, nodes),
                        values,
                        requirements,
                        constants,
                        new MetalNegAnalysisInputs(context, traceProducer)));
        return new IndexingRoute(partition, oneHotIndices, analysis);
    }

    private static ScatterRoute scatterRoute(MetalDeviceContext context) {
        ValueId data = new ValueId(41_000);
        ValueId indices = new ValueId(41_001);
        ValueId updates = new ValueId(41_002);
        ValueId output = new ValueId(41_003);
        TensorDescriptor dataDescriptor =
                typedDescriptor(DataType.FLOAT32, Shape.of(2, 3));
        TensorDescriptor indexDescriptor =
                typedDescriptor(DataType.INT32, Shape.of(2, 2));
        TensorDescriptor updateDescriptor =
                typedDescriptor(DataType.FLOAT32, Shape.of(2, 2));
        CompiledNode node = new CompiledNode(
                new NodeId(41_000),
                new Operation(
                        AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(1, ScatterReduction.NONE)),
                List.of(data, indices, updates),
                List.of(output));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID, List.of(node.id()));
        List<GraphValue> values = List.of(
                new GraphValue(data, dataDescriptor),
                new GraphValue(indices, indexDescriptor),
                new GraphValue(updates, updateDescriptor),
                new GraphValue(output, dataDescriptor));
        List<LogicalMemoryRequirement> requirements = List.of(
                requirement(data, dataDescriptor, Optional.empty(), List.of(partition), false),
                requirement(indices, indexDescriptor,
                        Optional.empty(), List.of(partition), false),
                requirement(updates, updateDescriptor,
                        Optional.empty(), List.of(partition), false),
                requirement(output, dataDescriptor,
                        Optional.of(partition), List.of(), true));
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis =
                new MetalNegPartitionPreparer().analyze(new PrepareContext<>(
                        NumericalProfile.STRICT_IEEE,
                        new PartitionDag(partition, List.of(node)),
                        values,
                        requirements,
                        Map.of(),
                        new MetalNegAnalysisInputs(context)));
        return new ScatterRoute(partition, indices, analysis);
    }

    private static BackendPartitionAnalysis<MetalNegPreparationPlan> task0052MinRoute(
            MetalDeviceContext context) {
        return new MetalNegPartitionPreparer().analyze(task0052MinContext(context));
    }

    private static PrepareContext<MetalNegAnalysisInputs> task0052MinContext(
            MetalDeviceContext context) {
        TensorDescriptor descriptor = descriptor(Shape.of(2));
        ValueId left = new ValueId(10_010);
        ValueId right = new ValueId(10_011);
        ValueId target = new ValueId(10_012);
        CompiledNode node = new CompiledNode(
                new NodeId(10_010),
                new Operation(BinaryArithmeticKind.MIN, NoOperationAttrs.INSTANCE),
                List.of(left, right),
                List.of(target));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID, List.of(node.id()));
        return new PrepareContext<>(
                NumericalProfile.ACCELERATOR,
                new PartitionDag(partition, List.of(node)),
                List.of(
                        new GraphValue(left, descriptor),
                        new GraphValue(right, descriptor),
                        new GraphValue(target, descriptor)),
                List.of(
                        requirement(left, descriptor, Optional.empty(), List.of(partition), false),
                        requirement(right, descriptor, Optional.empty(), List.of(partition), false),
                        requirement(target, descriptor, Optional.of(partition), List.of(), true)),
                Map.of(),
                new MetalNegAnalysisInputs(context));
    }

    private static TensorDescriptor typedDescriptor(DataType dataType, Shape shape) {
        return new TensorDescriptor(
                dataType, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static SingleNegRoute singleNegRoute(
            MetalDeviceContext context, Shape shape, Optional<ScalarValue> splat) {
        return singleNegRoute(context, shape, splat, null, null);
    }

    private static SingleNegRoute singleNegRoute(
            MetalDeviceContext context,
            Shape shape,
            Optional<ScalarValue> splat,
            MetalTraceProducer traceProducer) {
        return singleNegRoute(context, shape, splat, traceProducer, null);
    }

    private static SingleNegRoute singleNegRoute(
            MetalDeviceContext context,
            Shape shape,
            Optional<ScalarValue> splat,
            MetalTraceProducer traceProducer,
            MetalPreparedRoute forcedRoute) {
        TensorDescriptor descriptor = descriptor(shape);
        ValueId feed = new ValueId(10_000);
        ValueId target = new ValueId(10_001);
        Operation neg = new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE);
        CompiledNode node = new CompiledNode(
                new NodeId(10_000), neg, List.of(feed), List.of(target));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID, List.of(node.id()));
        List<GraphValue> values = List.of(
                new GraphValue(feed, descriptor), new GraphValue(target, descriptor));
        List<LogicalMemoryRequirement> requirements = List.of(
                requirement(feed, descriptor, Optional.empty(), List.of(partition), false),
                requirement(target, descriptor, Optional.of(partition), List.of(), true));
        Map<ValueId, ScalarValue> constants = splat
                .<Map<ValueId, ScalarValue>>map(value -> Map.of(feed, value))
                .orElseGet(Map::of);
        var prepareContext = new PrepareContext<>(
                NumericalProfile.STRICT_IEEE,
                new PartitionDag(partition, List.of(node)),
                values,
                requirements,
                constants,
                new MetalNegAnalysisInputs(context, traceProducer));
        var preparer = new MetalNegPartitionPreparer();
        BackendPartitionAnalysis<MetalNegPreparationPlan> analysis = forcedRoute == null
                ? preparer.analyze(prepareContext)
                : preparer.analyzeForTesting(prepareContext, forcedRoute);
        return new SingleNegRoute(partition, feed, target, analysis);
    }

    private static CompiledNode negNode(long nodeId, ValueId feed, ValueId target) {
        return new CompiledNode(
                new NodeId(nodeId),
                new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                List.of(feed),
                List.of(target));
    }

    private static BackendPartitionAnalysis<MetalNegPreparationPlan> splatConsumerAnalysis(
            MetalDeviceContext context,
            PlannedPartition partition,
            CompiledNode node,
            ValueId feed,
            ValueId target,
            TensorDescriptor descriptor,
            List<PlannedPartition> consumers) {
        return new MetalNegPartitionPreparer().analyze(new PrepareContext<>(
                NumericalProfile.STRICT_IEEE,
                new PartitionDag(partition, List.of(node)),
                List.of(new GraphValue(feed, descriptor), new GraphValue(target, descriptor)),
                List.of(
                        requirement(feed, descriptor, Optional.empty(), consumers, false),
                        requirement(target, descriptor, Optional.of(partition), List.of(), true)),
                Map.of(feed, ScalarValue.float32(4.0f)),
                new MetalNegAnalysisInputs(context)));
    }

    private static void closeResources(BackendPartitionFinalizationResult finalized) {
        if (finalized == null) {
            return;
        }
        for (int index = finalized.resources().size() - 1; index >= 0; index--) {
            finalized.resources().get(index).close();
        }
    }

    private static PreparedExecution prepareSingleExecution(
            MetalDeviceContext context, Shape shape, Optional<ScalarValue> splat) {
        return prepareSingleExecution(context, singleNegRoute(context, shape, splat));
    }

    private static PreparedExecution prepareSingleExecution(
            MetalDeviceContext context, SingleNegRoute route) {
        FinalizationFixture assignment = finalization(route.analysis());
        BackendPartitionFinalizationResult finalized = new MetalNegPartitionFinalizer(context)
                .finalizePartition(assignment.finalization());
        try {
            var schedule = new MetalNegPreparedScheduleAssembler(
                    context, route.analysis().plan(), List.of(route.target()))
                    .assembleRoute(assignment.memoryPlan(),
                            new PreparedPartition(route.partition(), finalized.executable()),
                            assignment.preparedAssignments());
            return new PreparedExecution(
                    assignment.memoryPlan(), schedule, finalized.resources());
        } catch (RuntimeException | Error failure) {
            for (int index = finalized.resources().size() - 1; index >= 0; index--) {
                try {
                    finalized.resources().get(index).close();
                } catch (RuntimeException | Error cleanup) {
                    if (cleanup != failure) {
                        failure.addSuppressed(cleanup);
                    }
                }
            }
            throw failure;
        }
    }

    private static FinalizationFixture finalization(
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis) {
        return finalization(analysis, false);
    }

    private static FinalizationFixture finalization(
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis, boolean reversePlanOrder) {
        MetalNegPreparationPlan plan = analysis.plan();
        int count = plan.declarations().size();
        var entries = new ArrayList<PreparedMemoryPlan.BufferEntry>();
        var assignments = new ArrayList<PreparationResourceAssignment>();
        var preparedByIndex = new PreparedBufferAssignment[count];
        var slots = new BufferSlot[count];
        for (int planIndex = 0; planIndex < count; planIndex++) {
            slots[planIndex] = new BufferSlot(700 + planIndex);
            int declarationIndex = reversePlanOrder ? count - 1 - planIndex : planIndex;
            var declaration = plan.declarations().get(declarationIndex);
            entries.add(new PreparedMemoryPlan.BufferEntry(
                    slots[planIndex], declaration.byteSize(), declaration.byteAlignment()));
            preparedByIndex[planIndex] = new PreparedBufferAssignment(declaration.valueId(), slots[planIndex], planIndex, java.util.List.of(io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider.METAL_BACKEND_ID));
        }
        for (int declarationIndex = 0; declarationIndex < count; declarationIndex++) {
            int planIndex = reversePlanOrder ? count - 1 - declarationIndex : declarationIndex;
            assignments.add(new PreparationResourceAssignment.Buffer(plan.declarations().get(declarationIndex), slots[planIndex], planIndex, 0));
        }
        var workspaceEntries = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
        plan.addressWorkspace().ifPresent(requirement -> {
            WorkspaceSlot workspaceSlot = new WorkspaceSlot(800);
            workspaceEntries.add(new PreparedMemoryPlan.WorkspaceEntry(workspaceSlot,
                    requirement.byteSize(), requirement.byteAlignment()));
            assignments.add(new PreparationResourceAssignment.Workspace(
                    requirement, workspaceSlot, 0));
        });
        var memoryPlan = new PreparedMemoryPlan(entries, workspaceEntries);
        return new FinalizationFixture(
                new BackendPartitionFinalization<>(analysis, memoryPlan, assignments),
                memoryPlan, List.copyOf(assignments), List.of(preparedByIndex));
    }

    private static void assertNativeFailure(
            MetalNativeApi.NativeFailure failure, String operation, int status) {
        assertEquals(operation, failure.operation());
        assertEquals(status, failure.statusCode());
        assertEquals(MetalNativeApi.Status.fromCode(status), failure.status());
    }

    private static void assertBindingRejected(
            MetalNegPreparedExecutable executable,
            PreparedMemoryPlan memoryPlan,
            MetalDeviceContext context,
            MetalDeviceContext other,
            BindingDefect defect,
            RecordingNativeApi api) {
        var buffers = new ArrayList<MetalBufferRepresentation>();
        RunState state = null;
        try {
            for (int index = 0; index < memoryPlan.buffers().size(); index++) {
                long bytes = memoryPlan.buffers().get(index).byteSize();
                MetalDeviceContext owner = index == 0 && defect == BindingDefect.WRONG_BUFFER_CONTEXT
                        ? other : context;
                long allocated = index == 0 && defect == BindingDefect.UNDERSIZED_BUFFER
                        ? bytes - 1 : bytes;
                buffers.add(owner.createBuffer(allocated));
            }
            if (defect == BindingDefect.CLOSED_BUFFER) buffers.getFirst().close();
            var representations = new ArrayList<>(buffers);
            if (defect == BindingDefect.OUTPUT_ALIAS) {
                representations.set(3, representations.get(2));
            }
            var bindings = representations.stream()
                    .map(buffer -> List.of(new BufferRepresentationBinding(
                            buffer, RunResourceOwnership.BORROWED)))
                    .toList();
            MetalDeviceContext workspaceContext = defect == BindingDefect.WRONG_WORKSPACE_CONTEXT
                    ? other : context;
            int pointerCount = defect == BindingDefect.WRONG_WORKSPACE_COUNT
                    ? executable.bufferSelectionCount() - 1
                    : executable.bufferSelectionCount();
            var workspace = new MetalNegPreparedExecutable.AddressWorkspace(
                    workspaceContext, pointerCount);
            if (defect == BindingDefect.CLOSED_WORKSPACE) workspace.close();
            if (defect == BindingDefect.OUTPUT_ALIAS) {
                assertThrows(IllegalArgumentException.class,
                        () -> new RunState(memoryPlan, bindings, List.of(workspace)));
                workspace.close();
                assertEquals(0, api.runCalls.get());
                return;
            }
            state = new RunState(memoryPlan, bindings, List.of(workspace));
            RunState boundState = state;
            assertThrows(IllegalArgumentException.class, () -> executable.bind(boundState));
            assertEquals(0, api.runCalls.get());
        } finally {
            if (state != null) state.close();
            buffers.forEach(MetalBufferRepresentation::close);
        }
    }

    private static void assertCustomBindingRejected(
            MetalNegPreparedExecutable executable,
            PreparedMemoryPlan memoryPlan,
            MetalBufferRepresentation input,
            MetalBufferRepresentation output) {
        var inputBinding = new BufferRepresentationBinding(
                input, RunResourceOwnership.BORROWED);
        var outputBinding = new BufferRepresentationBinding(
                output, RunResourceOwnership.BORROWED);
        assertThrows(IllegalArgumentException.class, () -> {
            try (RunState state = new RunState(
                    memoryPlan,
                    List.of(List.of(inputBinding), List.of(outputBinding)),
                    List.of())) {
                executable.bind(state);
            }
        });
    }

    private static Fixture fixture() {
        return fixture(Shape.of(4));
    }

    private static Fixture fixture(Shape secondShape) {
        TensorDescriptor a = descriptor(Shape.of(2, 3));
        TensorDescriptor b = descriptor(secondShape);
        ValueId v0 = new ValueId(0), v1 = new ValueId(1), v2 = new ValueId(2);
        ValueId v3 = new ValueId(3), v4 = new ValueId(4), v5 = new ValueId(5);
        Operation neg = new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE);
        List<CompiledNode> nodes = List.of(
                new CompiledNode(new NodeId(0), neg, List.of(v0), List.of(v2)),
                new CompiledNode(new NodeId(1), neg, List.of(v2), List.of(v3)),
                new CompiledNode(new NodeId(2), neg, List.of(v1), List.of(v4)),
                new CompiledNode(new NodeId(3), neg, List.of(v3), List.of(v5)));
        PlannedPartition partition = new PlannedPartition(MetalCapabilityProvider.METAL_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        List<GraphValue> values = List.of(new GraphValue(v0, a), new GraphValue(v1, b),
                new GraphValue(v2, a), new GraphValue(v3, a), new GraphValue(v4, b),
                new GraphValue(v5, a));
        List<LogicalMemoryRequirement> requirements = List.of(
                requirement(v0, a, Optional.empty(), List.of(partition), false),
                requirement(v1, b, Optional.empty(), List.of(partition), false),
                requirement(v2, a, Optional.of(partition), List.of(partition), false),
                requirement(v3, a, Optional.of(partition), List.of(partition), true),
                requirement(v4, b, Optional.of(partition), List.of(), true),
                requirement(v5, a, Optional.of(partition), List.of(), true));
        return new Fixture(partition, nodes, values, requirements, v0, v1, v2, v3, v4, v5);
    }

    private static Fixture withRequirement(
            Fixture fixture, LogicalMemoryRequirement replacement) {
        var requirements = new ArrayList<>(fixture.requirements());
        for (int index = 0; index < requirements.size(); index++) {
            if (requirements.get(index).valueId().equals(replacement.valueId())) {
                requirements.set(index, replacement);
                return new Fixture(fixture.partition(), fixture.nodes(), fixture.values(),
                        requirements, fixture.v0(), fixture.v1(), fixture.v2(), fixture.v3(),
                        fixture.v4(), fixture.v5());
            }
        }
        throw new AssertionError("missing fixture requirement " + replacement.valueId());
    }

    private static TensorDescriptor descriptorFor(Fixture fixture, ValueId valueId) {
        return fixture.values().stream()
                .filter(value -> value.id().equals(valueId))
                .findFirst()
                .orElseThrow()
                .descriptor();
    }

    private static LogicalMemoryRequirement requirement(ValueId id, TensorDescriptor descriptor,
            Optional<PlannedPartition> producer, List<PlannedPartition> consumers, boolean output) {
        return new LogicalMemoryRequirement(id, descriptor, producer, consumers, output);
    }

    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static void close(AutoCloseable closeable) {
        if (closeable == null) return;
        try { closeable.close(); } catch (Exception failure) { throw new AssertionError(failure); }
    }

    private record Fixture(
            PlannedPartition partition, List<CompiledNode> nodes, List<GraphValue> values,
            List<LogicalMemoryRequirement> requirements, ValueId v0, ValueId v1, ValueId v2,
            ValueId v3, ValueId v4, ValueId v5) { }
    private record AffineFixture(
            PlannedPartition partition,
            List<CompiledNode> nodes,
            List<GraphValue> values,
            List<LogicalMemoryRequirement> requirements,
            List<ValueId> feeds,
            List<ValueId> targets) { }


    private record SplatRoute(
            MetalDeviceContext context,
            PreparedExecution execution,
            MetalNegPreparationPlan plan,
            int[] splatBits,
            int targetCount) { }

    private record IndexingRoute(
            PlannedPartition partition,
            ValueId oneHotIndices,
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis) { }

    private record ScatterRoute(
            PlannedPartition partition,
            ValueId indices,
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis) { }

    private record SingleNegRoute(
            PlannedPartition partition,
            ValueId feed,
            ValueId target,
            BackendPartitionAnalysis<MetalNegPreparationPlan> analysis) { }

    private record FinalizationFixture(
            BackendPartitionFinalization<MetalNegPreparationPlan> finalization,
            PreparedMemoryPlan memoryPlan,
            List<PreparationResourceAssignment> assignments,
            List<PreparedBufferAssignment> preparedAssignments) { }

    private enum BindingDefect {
        WRONG_BUFFER_CONTEXT,
        CLOSED_BUFFER,
        UNDERSIZED_BUFFER,
        OUTPUT_ALIAS,
        WRONG_WORKSPACE_CONTEXT,
        WRONG_WORKSPACE_COUNT,
        CLOSED_WORKSPACE
    }

    private static final class RecordingNativeApi extends MetalNativeApi {
        private long next = 1;
        private final AtomicInteger executableCreates = new AtomicInteger();
        private final AtomicInteger executableReleases = new AtomicInteger();
        private final AtomicInteger pipelineCreates = new AtomicInteger();
        private final AtomicInteger pipelineReleases = new AtomicInteger();
        private final AtomicInteger customRunCalls = new AtomicInteger();
        private final AtomicInteger contextReleases = new AtomicInteger();
        private final AtomicInteger apiCloses = new AtomicInteger();
        private final AtomicInteger runCalls = new AtomicInteger();
        private final AtomicInteger bufferCreates = new AtomicInteger();
        private final AtomicInteger uploads = new AtomicInteger();
        private final AtomicInteger bufferReleases = new AtomicInteger();
        private final Map<Long, byte[]> buffers = new ConcurrentHashMap<>();
        private final Map<Thread, List<Long>> createdHandlesByThread =
                new ConcurrentHashMap<>();
        private final List<RunObservation> runs = new CopyOnWriteArrayList<>();
        private final List<Long> customInputHandles = new CopyOnWriteArrayList<>();
        private final Queue<RuntimeException> bufferReleaseFailures =
                new ConcurrentLinkedQueue<>();
        private RuntimeException createFailure;
        private RuntimeException executableReleaseFailure;
        private RuntimeException contextReleaseFailure;
        private RuntimeException apiCloseFailure;
        private RuntimeException bufferCreateFailure;
        private RuntimeException uploadFailure;
        private volatile int createStatus;
        private volatile int runStatus;
        private volatile int releaseStatus;
        private volatile int pipelineCreateStatus;
        private volatile int pipelineReleaseStatus;
        private volatile int customRunStatus;
        private volatile boolean createNullHandle;
        private volatile boolean createHandleOnFailure;
        private volatile boolean emulateSingletonMpsGraphNeg;
        private volatile int failBufferCreateCall = -1;
        private volatile int failUploadCall = -1;
        private volatile CountDownLatch runEntered = new CountDownLatch(0);
        private volatile CountDownLatch continueRuns = new CountDownLatch(0);
        private volatile CountDownLatch bufferCreateEntered = new CountDownLatch(0);
        private byte[] createdProgramImage;

        @Override synchronized Handle createContext() { return handle(); }
        @Override void releaseContext(Handle context) {
            contextReleases.incrementAndGet();
            if (contextReleaseFailure != null) throw contextReleaseFailure;
        }
        @Override synchronized Handle createBuffer(Handle context, long bytes) {
            int call = bufferCreates.incrementAndGet();
            if (call == failBufferCreateCall) throw bufferCreateFailure;
            Handle handle = handle();
            buffers.put(handle.carrier().address(), new byte[Math.toIntExact(bytes)]);
            createdHandlesByThread.computeIfAbsent(
                    Thread.currentThread(), ignored -> new CopyOnWriteArrayList<>())
                    .add(handle.carrier().address());
            bufferCreateEntered.countDown();
            return handle;
        }
        @Override void releaseBuffer(Handle buffer) {
            bufferReleases.incrementAndGet();
            buffers.remove(buffer.carrier().address());
            RuntimeException failure = bufferReleaseFailures.poll();
            if (failure != null) throw failure;
        }
        @Override void upload(Handle buffer, long offset, MemorySegment source, long count) {
            int call = uploads.incrementAndGet();
            if (call == failUploadCall) throw uploadFailure;
            byte[] target = buffers.get(buffer.carrier().address());
            for (long index = 0; index < count; index++) {
                target[Math.toIntExact(offset + index)] = source.getAtIndex(JAVA_BYTE, index);
            }
        }
        @Override void download(Handle buffer, long offset, MemorySegment destination, long count) {
            byte[] source = buffers.get(buffer.carrier().address());
            for (long index = 0; index < count; index++) {
                destination.setAtIndex(JAVA_BYTE, index,
                        source[Math.toIntExact(offset + index)]);
            }
        }
        @Override
        synchronized NativeCreateResult createMpsGraphExecutableNative(
                Handle context,
                MemorySegment programImage) {
            executableCreates.incrementAndGet();
            if (createFailure != null) throw createFailure;
            createdProgramImage = programImage.toArray(JAVA_BYTE);
            Handle created = createNullHandle ? null : handle();
            if (createStatus != 0 && !createHandleOnFailure) created = null;
            return new NativeCreateResult(createStatus, created);
        }
        @Override int releaseExecutableNative(Handle executable) {
            executableReleases.incrementAndGet();
            if (executableReleaseFailure != null) throw executableReleaseFailure;
            return releaseStatus;
        }
        @Override int runExecutableNative(Handle executable, int inputCount, MemorySegment inputs,
                int outputCount, MemorySegment outputs) {
            runCalls.incrementAndGet();
            var inputHandles = addresses(inputs, inputCount);
            var outputHandles = addresses(outputs, outputCount);
            var inputRawBits = new ArrayList<int[]>(inputHandles.size());
            for (long input : inputHandles) {
                byte[] bytes = buffers.get(input);
                int[] bits = new int[bytes.length / Integer.BYTES];
                MemorySegment segment = MemorySegment.ofArray(bytes);
                for (int index = 0; index < bits.length; index++) {
                    bits[index] = segment.getAtIndex(JAVA_INT_UNALIGNED, index);
                }
                inputRawBits.add(bits);
            }
            if (emulateSingletonMpsGraphNeg && runStatus == 0) {
                if (inputHandles.size() != 1 || outputHandles.size() != 1) {
                    throw new IllegalStateException("singleton NEG emulator requires one input/output");
                }
                byte[] input = buffers.get(inputHandles.getFirst());
                byte[] output = buffers.get(outputHandles.getFirst());
                if (input == null || output == null || input.length > output.length) {
                    throw new IllegalStateException("singleton NEG emulator buffer mismatch");
                }
                MemorySegment inputSegment = MemorySegment.ofArray(input);
                MemorySegment outputSegment = MemorySegment.ofArray(output);
                for (long index = 0; index < input.length / Integer.BYTES; index++) {
                    int bits = inputSegment.getAtIndex(JAVA_INT_UNALIGNED, index);
                    outputSegment.setAtIndex(
                            JAVA_INT_UNALIGNED, index, bits ^ 0x8000_0000);
                }
            }
            runs.add(new RunObservation(inputHandles, outputHandles, inputs, inputRawBits,
                    bufferCreates.get(), uploads.get()));
            runEntered.countDown();
            try {
                if (!continueRuns.await(5, TimeUnit.SECONDS)) {
                    throw new IllegalStateException("timed out waiting to continue Metal run");
                }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException(interrupted);
            }
            return runStatus;
        }
        @Override synchronized NativeCreateResult createNegKernelPipelineNative(
                Handle context, long elementCount) {
            pipelineCreates.incrementAndGet();
            if (createFailure != null) throw createFailure;
            if (elementCount == 0L || elementCount > 0xffff_ffffL) {
                return new NativeCreateResult(8, null);
            }
            Handle created = createNullHandle ? null : handle();
            if (pipelineCreateStatus != 0 && !createHandleOnFailure) created = null;
            return new NativeCreateResult(pipelineCreateStatus, created);
        }
        @Override int releaseNegKernelPipelineNative(Handle pipeline) {
            pipelineReleases.incrementAndGet();
            if (executableReleaseFailure != null) throw executableReleaseFailure;
            return pipelineReleaseStatus;
        }
        @Override int runNegKernelPipelineNative(
                Handle pipeline, Handle inputBuffer, Handle outputBuffer) {
            customRunCalls.incrementAndGet();
            customInputHandles.add(inputBuffer.carrier().address());
            byte[] input = buffers.get(inputBuffer.carrier().address());
            byte[] output = buffers.get(outputBuffer.carrier().address());
            if (input == null || output == null || input.length > output.length) {
                return 10;
            }
            MemorySegment inputSegment = MemorySegment.ofArray(input);
            MemorySegment outputSegment = MemorySegment.ofArray(output);
            for (long index = 0; index < input.length / Integer.BYTES; index++) {
                int bits = inputSegment.getAtIndex(JAVA_INT_UNALIGNED, index);
                outputSegment.setAtIndex(JAVA_INT_UNALIGNED, index, bits ^ 0x8000_0000);
            }
            runEntered.countDown();
            try {
                if (!continueRuns.await(5, TimeUnit.SECONDS)) {
                    throw new IllegalStateException("timed out waiting to continue custom run");
                }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException(interrupted);
            }
            return customRunStatus;
        }
        @Override public void close() {
            apiCloses.incrementAndGet();
            if (apiCloseFailure != null) throw apiCloseFailure;
        }
        private Handle handle() { return new Handle(MemorySegment.ofAddress(next++)); }

        private void blockRuns(int count) {
            runEntered = new CountDownLatch(count);
            continueRuns = new CountDownLatch(1);
        }

        private void blockBufferCreates(int count) {
            createdHandlesByThread.clear();
            bufferCreateEntered = new CountDownLatch(count);
        }

        private List<List<Long>> concurrentCreatedHandles() {
            return createdHandlesByThread.values().stream()
                    .map(List::copyOf)
                    .toList();
        }

        private Set<Long> liveBufferHandles() {
            return Set.copyOf(buffers.keySet());
        }

        private static List<Long> addresses(MemorySegment segment, int count) {
            var result = new ArrayList<Long>(count);
            for (int index = 0; index < count; index++) {
                result.add(segment.getAtIndex(ADDRESS, index).address());
            }
            return List.copyOf(result);
        }

        private record RunObservation(
                List<Long> inputHandles,
                List<Long> outputHandles,
                MemorySegment inputAddresses,
                List<int[]> inputRawBits,
                int bufferCreateCount,
                int uploadCount) {
            List<Long> splatHandles() {
                return inputHandles.subList(1, inputHandles.size());
            }
        }
    }
}
