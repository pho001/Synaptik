package io.github.pho001.synaptik.backend.cpu.execution;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.*;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable.BufferAccess;
import java.lang.classfile.ClassFile;
import java.nio.ByteOrder;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class CpuPortablePartitionPreparerTest {
    static final CpuLoweringFingerprint LOWERING = CpuLoweringFingerprint.of("cpu-0004".getBytes());
    static final CpuKernelSpecialization.VectorShape VECTOR_SHAPE =
            new CpuKernelSpecialization.VectorShape(DataType.FLOAT32, 128, 4);

    @Test
    void validatesImmutableInputsAndCandidateShape() {
        assertEquals("workerCount must be positive", assertThrows(IllegalArgumentException.class,
                () -> new CpuPreparedParallelConfiguration(0, 1, true)).getMessage());
        assertEquals("minimumRangeSize must be positive", assertThrows(IllegalArgumentException.class,
                () -> new CpuPreparedParallelConfiguration(1, 0, true)).getMessage());
        var shapes = new ArrayList<>(List.of(VECTOR_SHAPE));
        var inputs = new CpuPortableAnalysisInputs(
                shapes, new CpuPreparedParallelConfiguration(2, 4, true));
        shapes.clear();
        assertEquals(List.of(VECTOR_SHAPE), inputs.supportedVectorShapes());
        assertEquals("supportedVectorShapes[1] duplicates " + VECTOR_SHAPE,
                assertThrows(IllegalArgumentException.class,
                        () -> new CpuPortableAnalysisInputs(List.of(VECTOR_SHAPE, VECTOR_SHAPE),
                                new CpuPreparedParallelConfiguration(1, 1, true))).getMessage());
        assertEquals("supportedVectorShapes", assertThrows(NullPointerException.class,
                () -> new CpuPortableAnalysisInputs(null,
                        new CpuPreparedParallelConfiguration(1, 1, true))).getMessage());
        assertEquals("supportedVectorShapes[0]", assertThrows(NullPointerException.class,
                () -> new CpuPortableAnalysisInputs(
                        Arrays.asList((CpuKernelSpecialization.VectorShape) null),
                        new CpuPreparedParallelConfiguration(1, 1, true))).getMessage());
        assertEquals("parallelConfiguration", assertThrows(NullPointerException.class,
                () -> new CpuPortableAnalysisInputs(List.of(), null)).getMessage());
        assertAll(
                () -> assertTrue(ModifierAssertions.packagePrivate(CpuPreparedParallelConfiguration.class)),
                () -> assertTrue(ModifierAssertions.packagePrivate(CpuPortableAnalysisInputs.class)),
                () -> assertTrue(ModifierAssertions.packagePrivate(CpuPortableCandidateSource.class)),
                () -> assertTrue(ModifierAssertions.packagePrivateFinal(CpuPortableKernelCandidate.class)),
                () -> assertTrue(ModifierAssertions.packagePrivateFinal(CpuPortablePartitionCandidate.class)),
                () -> assertTrue(ModifierAssertions.packagePrivate(CpuPortableInvocationBinder.class)),
                () -> assertTrue(ModifierAssertions.packagePrivateFinal(CpuPortablePreparationPlan.class)),
                () -> assertTrue(ModifierAssertions.packagePrivateFinal(CpuPortablePartitionPreparer.class)),
                () -> assertTrue(ModifierAssertions.packagePrivateFinal(CpuPortablePartitionFinalizer.class)),
                () -> assertTrue(ModifierAssertions.packagePrivateFinal(CpuPortablePreparedExecutable.class)));
        assertExactSurface();
    }

    @Test
    void candidateValidatesNullsSnapshotsFingerprintsDeclarationsAndUsesInStableOrder() {
        var specialization = specialization(CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                List.of(argument(CpuKernelSpecialization.Carrier.FLOAT_ARRAY)));
        var requirement = new PreparationResourceRequirement.Buffer(new ValueId(0), 16, 4);
        var use = new CpuPortableKernelCandidate.BufferUse(requirement, 3);
        var binder = (CpuPortableInvocationBinder)
                (state, handle, spec, parallel, workers, buffers, workspaces) -> null;
        assertEquals("requirement", assertThrows(NullPointerException.class,
                () -> new CpuPortableKernelCandidate.BufferUse(null, 0)).getMessage());
        assertEquals("representationIndex must be non-negative",
                assertThrows(IllegalArgumentException.class,
                        () -> new CpuPortableKernelCandidate.BufferUse(requirement, -1)).getMessage());
        assertEquals("requirement", assertThrows(NullPointerException.class,
                () -> new CpuPortableKernelCandidate.WorkspaceUse(null)).getMessage());
        assertEquals("specialization", candidateFailure(null, emitter(), List.of(requirement),
                List.of(use), List.of(), binder).getMessage());
        assertEquals("familyEmitter", candidateFailure(specialization, null,
                List.of(requirement), List.of(use), List.of(), binder).getMessage());
        CpuFamilyKernelEmitter nullFingerprint = new CpuFamilyKernelEmitter() {
            @Override public CpuLoweringFingerprint loweringFingerprint() { return null; }
            @Override public void emitScalar(CpuScalarEmitter s, CpuCarrierEmitter c,
                    CpuLoopEmitter l, CpuReductionEmitter r) { }
            @Override public void emitVector(CpuVectorEmitter v, CpuCarrierEmitter c,
                    CpuLoopEmitter l, CpuReductionEmitter r) { }
        };
        assertEquals("familyEmitter.loweringFingerprint()", candidateFailure(specialization,
                nullFingerprint, List.of(requirement), List.of(use), List.of(), binder).getMessage());
        assertEquals("familyEmitter lowering fingerprint does not match specialization",
                candidateFailure(specialization,
                        emitter(CpuLoweringFingerprint.of("different".getBytes())),
                        List.of(requirement), List.of(use), List.of(), binder).getMessage());
        assertEquals("requirements", candidateFailure(specialization, emitter(), null,
                List.of(use), List.of(), binder).getMessage());
        assertEquals("requirements[0]", candidateFailure(specialization, emitter(),
                Arrays.asList((PreparationResourceRequirement) null), List.of(use), List.of(), binder)
                .getMessage());
        var duplicate = new PreparationResourceRequirement.Buffer(new ValueId(0), 16, 8);
        assertTrue(candidateFailure(specialization, emitter(), List.of(requirement, duplicate),
                List.of(use), List.of(), binder).getMessage().contains("duplicates buffer"));
        var workspace = new PreparationResourceRequirement.Workspace(7, 8, 8);
        var workspaceDuplicate = new PreparationResourceRequirement.Workspace(7, 16, 16);
        assertTrue(candidateFailure(specialization, emitter(),
                List.of(requirement, workspace, workspaceDuplicate), List.of(use),
                List.of(new CpuPortableKernelCandidate.WorkspaceUse(workspace)), binder)
                .getMessage().contains("duplicates workspace"));
        assertEquals("bufferUses", candidateFailure(specialization, emitter(),
                List.of(requirement), null, List.of(), binder).getMessage());
        assertEquals("bufferUses[0]", candidateFailure(specialization, emitter(),
                List.of(requirement), Arrays.asList((CpuPortableKernelCandidate.BufferUse) null),
                List.of(), binder).getMessage());
        assertEquals("workspaceUses", candidateFailure(specialization, emitter(),
                List.of(requirement), List.of(use), null, binder).getMessage());
        assertEquals("workspaceUses[0]", candidateFailure(specialization, emitter(),
                List.of(requirement), List.of(use),
                Arrays.asList((CpuPortableKernelCandidate.WorkspaceUse) null), binder).getMessage());
        assertEquals("invocationBinder", candidateFailure(specialization, emitter(),
                List.of(requirement), List.of(use), List.of(), null).getMessage());
        var equalButDistinct = new PreparationResourceRequirement.Buffer(new ValueId(0), 16, 4);
        assertEquals("bufferUses[0].requirement is not declared", candidateFailure(specialization,
                emitter(), List.of(requirement),
                List.of(new CpuPortableKernelCandidate.BufferUse(equalButDistinct, 0)),
                List.of(), binder).getMessage());
        assertEquals("requirements[1] is unused", candidateFailure(specialization, emitter(),
                List.of(requirement, workspace), List.of(use), List.of(), binder).getMessage());
        assertEquals("bufferUses size must equal specialization argument count 1",
                candidateFailure(specialization, emitter(), List.of(requirement), List.of(),
                        List.of(), binder).getMessage());

        var requirements = new ArrayList<PreparationResourceRequirement>(List.of(requirement));
        var uses = new ArrayList<>(List.of(use));
        var candidate = new CpuPortableKernelCandidate(specialization, emitter(), requirements,
                uses, List.of(), binder);
        requirements.clear(); uses.clear();
        assertAll(() -> assertEquals(List.of(requirement), candidate.requirements()),
                () -> assertEquals(List.of(use), candidate.bufferUses()),
                () -> assertEquals(3, candidate.bufferUses().getFirst().representationIndex()),
                () -> assertSame(binder, candidate.invocationBinder()));

        var workspaceUses = new ArrayList<>(
                List.of(new CpuPortableKernelCandidate.WorkspaceUse(workspace)));
        var withWorkspace = new CpuPortableKernelCandidate(specialization, emitter(),
                List.of(requirement, workspace), List.of(use), workspaceUses, binder);
        workspaceUses.clear();
        assertEquals(1, withWorkspace.workspaceUses().size());
        assertSame(workspace, withWorkspace.workspaceUses().getFirst().requirement());

        var configuration = new CpuPreparedParallelConfiguration(1, 1, true);
        assertEquals("candidate", assertThrows(NullPointerException.class,
                () -> new CpuPortablePreparationPlan(
                        (CpuPortablePartitionCandidate) null, configuration)).getMessage());
        assertEquals("parallelConfiguration", assertThrows(NullPointerException.class,
                () -> new CpuPortablePreparationPlan(partition(candidate), null)).getMessage());
    }

    @Test
    void rejectsWrongOwnerBeforeCallingSourceAndSelectsFirstEligibleCandidate() {
        var calls = new int[1];
        var preparer = new CpuPortablePartitionPreparer(context -> {
            calls[0]++;
            return List.of(partition(candidate(CpuPortableExecutionMode.VECTOR_API_SINGLE_THREAD)),
                    partition(candidate(CpuPortableExecutionMode.SCALAR_SINGLE_THREAD)));
        });
        var wrong = context(new BackendId("other"), List.of());
        assertEquals("partition owner must be CPU",
                assertThrows(IllegalArgumentException.class, () -> preparer.analyze(wrong)).getMessage());
        assertEquals(0, calls[0]);

        var context = context(CpuCapabilityProvider.CPU_BACKEND_ID, List.of());
        var analysis = preparer.analyze(context);
        assertSame(context.partition(), analysis.partition());
        assertSame(CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                analysis.plan().candidate().specialization().executionMode());
        assertSame(analysis.plan().candidate().requirements().getFirst(),
                analysis.requirements().getFirst());
    }

    @Test
    void failsClosedForEmptyInvalidAndUnprojectedCandidates() {
        var context = context(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(VECTOR_SHAPE));
        assertEquals("no supported CPU portable candidate", assertThrows(IllegalArgumentException.class,
                () -> new CpuPortablePartitionPreparer(ignored -> List.of()).analyze(context)).getMessage());
        assertEquals("candidateSource", assertThrows(NullPointerException.class,
                () -> new CpuPortablePartitionPreparer(null)).getMessage());
        assertEquals("context", assertThrows(NullPointerException.class,
                () -> new CpuPortablePartitionPreparer(ignored -> List.of()).analyze(null)).getMessage());
        assertEquals("candidates", assertThrows(NullPointerException.class,
                () -> new CpuPortablePartitionPreparer(ignored -> null).analyze(context)).getMessage());
        assertEquals("candidates[0]", assertThrows(NullPointerException.class,
                () -> new CpuPortablePartitionPreparer(ignored -> Arrays.asList((CpuPortablePartitionCandidate) null))
                        .analyze(context)).getMessage());
        var foreignRequirement = new PreparationResourceRequirement.Buffer(new ValueId(99), 16, 4);
        var specialization = specialization(CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                List.of(argument(CpuKernelSpecialization.Carrier.FLOAT_ARRAY)));
        var malformed = new CpuPortableKernelCandidate(specialization, emitter(),
                List.of(foreignRequirement),
                List.of(new CpuPortableKernelCandidate.BufferUse(foreignRequirement, 0)),
                List.of(), (state, handle, spec, parallel, workers, buffers, workspaces) -> null);
        assertTrue(assertThrows(IllegalArgumentException.class,
                () -> new CpuPortablePartitionPreparer(ignored -> List.of(partition(malformed))).analyze(context))
                .getMessage().contains("value is not projected"));
    }

    @Test
    void validatesTypeButPreservesOpaqueCandidateByteGeometryAndHasNoAnalysisSideEffects() {
        var context = context(CpuCapabilityProvider.CPU_BACKEND_ID, List.of());
        var wrongTypeArgument = new CpuKernelSpecialization.Argument(DataType.INT32,
                CpuKernelSpecialization.Carrier.INT_ARRAY, BufferAccess.READ_ONLY,
                true, 0, List.of(1L));
        var typeRequirement = new PreparationResourceRequirement.Buffer(new ValueId(0), 16, 4);
        var wrongType = candidate(specialization(CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                List.of(wrongTypeArgument)), List.of(typeRequirement),
                List.of(new CpuPortableKernelCandidate.BufferUse(typeRequirement, 0)), List.of(),
                emitter(), (state, handle, spec, parallel, workers, buffers, workspaces) -> null);
        assertEquals("candidates[0].bufferUses[0] data type does not match specialization argument",
                assertThrows(IllegalArgumentException.class,
                        () -> new CpuPortablePartitionPreparer(ignored -> List.of(partition(wrongType)))
                                .analyze(context)).getMessage());

        var emissions = new AtomicInteger();
        var binds = new AtomicInteger();
        var opaqueRequirement = new PreparationResourceRequirement.Buffer(new ValueId(0), 12, 4);
        var countingEmitter = countingEmitter(emissions);
        var opaque = candidate(specialization(CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                        List.of(argument(CpuKernelSpecialization.Carrier.FLOAT_ARRAY))),
                List.of(opaqueRequirement),
                List.of(new CpuPortableKernelCandidate.BufferUse(opaqueRequirement, 0)), List.of(),
                countingEmitter, (state, handle, spec, parallel, workers, buffers, workspaces) -> {
                    binds.incrementAndGet(); return null;
                });
        var analysis = new CpuPortablePartitionPreparer(ignored -> List.of(partition(opaque))).analyze(context);
        assertAll(() -> assertSame(opaqueRequirement, analysis.requirements().getFirst()),
                () -> assertEquals(12, ((PreparationResourceRequirement.Buffer)
                        analysis.requirements().getFirst()).byteSize()),
                () -> assertEquals(0, emissions.get()), () -> assertEquals(0, binds.get()));
    }

    @Test
    void analyzesAllModesRepeatedUsesAndConcurrentCallsInDeterministicSourceOrder()
            throws Exception {
        for (var mode : CpuPortableExecutionMode.values()) {
            var shapes = mode.vectorized() ? List.of(VECTOR_SHAPE)
                    : List.<CpuKernelSpecialization.VectorShape>of();
            var analysis = new CpuPortablePartitionPreparer(ignored -> List.of(partition(candidate(mode))))
                    .analyze(context(CpuCapabilityProvider.CPU_BACKEND_ID, shapes));
            assertSame(mode, analysis.plan().candidate().specialization().executionMode());
            assertEquals(2, analysis.plan().parallelConfiguration().workerCount());
        }
        assertEquals("no supported CPU portable candidate", assertThrows(IllegalArgumentException.class,
                () -> new CpuPortablePartitionPreparer(ignored ->
                        List.of(partition(candidate(CpuPortableExecutionMode.VECTOR_API_PARALLEL))))
                        .analyze(context(CpuCapabilityProvider.CPU_BACKEND_ID, List.of())))
                .getMessage());
        var requirement = new PreparationResourceRequirement.Buffer(new ValueId(0), 16, 4);
        var repeated = candidate(specialization(CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                        List.of(argument(CpuKernelSpecialization.Carrier.FLOAT_ARRAY),
                                argument(CpuKernelSpecialization.Carrier.FLOAT_ARRAY))),
                List.of(requirement), List.of(
                        new CpuPortableKernelCandidate.BufferUse(requirement, 0),
                        new CpuPortableKernelCandidate.BufferUse(requirement, 2)), List.of(),
                emitter(), (state, handle, spec, parallel, workers, buffers, workspaces) -> null);
        var repeatedAnalysis = new CpuPortablePartitionPreparer(ignored -> List.of(partition(repeated)))
                .analyze(context(CpuCapabilityProvider.CPU_BACKEND_ID, List.of()));
        assertAll(() -> assertSame(requirement, repeatedAnalysis.requirements().getFirst()),
                () -> assertEquals(0, repeated.bufferUses().get(0).representationIndex()),
                () -> assertEquals(2, repeated.bufferUses().get(1).representationIndex()),
                () -> assertEquals(BufferAccess.READ_ONLY,
                        repeated.specialization().arguments().get(1).access()),
                () -> assertEquals(DataType.FLOAT32,
                        repeated.specialization().arguments().get(1).dataType()),
                () -> assertEquals(CpuKernelSpecialization.Carrier.FLOAT_ARRAY,
                        repeated.specialization().arguments().get(1).carrier()));
        var calls = new AtomicInteger();
        var preparer = new CpuPortablePartitionPreparer(ignored -> {
            calls.incrementAndGet(); return List.of(partition(
                    candidate(CpuPortableExecutionMode.SCALAR_SINGLE_THREAD)));
        });
        var shared = context(CpuCapabilityProvider.CPU_BACKEND_ID, List.of());
        try (var executor = Executors.newFixedThreadPool(4)) {
            var futures = new ArrayList<java.util.concurrent.Future<BackendPartitionAnalysis<CpuPortablePreparationPlan>>>();
            for (int index = 0; index < 16; index++) futures.add(executor.submit(() -> preparer.analyze(shared)));
            for (var future : futures) assertSame(shared.partition(), future.get().partition());
        }
        assertEquals(16, calls.get());
    }

    static PrepareContext<CpuPortableAnalysisInputs> context(
            BackendId owner, List<CpuKernelSpecialization.VectorShape> vectorShapes) {
        ValueId input = new ValueId(0);
        ValueId output = new ValueId(1);
        CompiledNode node = new CompiledNode(new NodeId(0),
                new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE),
                List.of(input), List.of(output));
        PlannedPartition partition = new PlannedPartition(owner, List.of(node.id()));
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, Shape.of(4), Optional.empty(), false);
        return new PrepareContext<>(partition, List.of(node),
                List.of(new GraphValue(input, descriptor), new GraphValue(output, descriptor)),
                List.of(
                        new LogicalMemoryRequirement(input, descriptor, Optional.empty(),
                                List.of(partition), false),
                        new LogicalMemoryRequirement(output, descriptor, Optional.of(partition),
                                List.of(), true)),
                Map.of(), new CpuPortableAnalysisInputs(vectorShapes,
                        new CpuPreparedParallelConfiguration(2, 1, true)));
    }

    static CpuPortableKernelCandidate candidate(CpuPortableExecutionMode mode) {
        var requirement = new PreparationResourceRequirement.Buffer(new ValueId(0), 16, 4);
        return candidate(specialization(mode,
                        List.of(argument(CpuKernelSpecialization.Carrier.FLOAT_ARRAY))),
                List.of(requirement),
                List.of(new CpuPortableKernelCandidate.BufferUse(requirement, 0)), List.of(),
                emitter(), (state, handle, spec, parallel, workers, buffers, workspaces) -> null);
    }

    static CpuPortablePartitionCandidate partition(CpuPortableKernelCandidate candidate) {
        return new CpuPortablePartitionCandidate(candidate.requirements(), List.of(candidate));
    }

    static CpuPortableKernelCandidate candidate(CpuKernelSpecialization specialization,
            List<PreparationResourceRequirement> requirements,
            List<CpuPortableKernelCandidate.BufferUse> bufferUses,
            List<CpuPortableKernelCandidate.WorkspaceUse> workspaceUses,
            CpuFamilyKernelEmitter familyEmitter, CpuPortableInvocationBinder binder) {
        return new CpuPortableKernelCandidate(specialization, familyEmitter, requirements,
                bufferUses, workspaceUses, binder);
    }

    static CpuKernelSpecialization specialization(
            CpuPortableExecutionMode mode, List<CpuKernelSpecialization.Argument> arguments) {
        return new CpuKernelSpecialization(CpuGeneratorSchema.CURRENT_VERSION, LOWERING, mode,
                arguments, List.of(), 0, mode.vectorized() ? VECTOR_SHAPE : null,
                ByteOrder.nativeOrder(), 1, 1, CpuKernelSpecialization.Tail.NONE,
                CpuKernelSpecialization.NumericalMode.EXACT_DEFAULT,
                CpuKernelSpecialization.CombineOrder.FIXED, ClassFile.JAVA_26_VERSION);
    }

    static CpuKernelSpecialization.Argument argument(CpuKernelSpecialization.Carrier carrier) {
        return new CpuKernelSpecialization.Argument(DataType.FLOAT32, carrier,
                BufferAccess.READ_ONLY, true, 0, List.of(1L));
    }

    static CpuFamilyKernelEmitter emitter() {
        return emitter(LOWERING);
    }

    static CpuFamilyKernelEmitter emitter(CpuLoweringFingerprint fingerprint) {
        return new CpuFamilyKernelEmitter() {
            @Override public CpuLoweringFingerprint loweringFingerprint() { return fingerprint; }
            @Override public void emitScalar(CpuScalarEmitter scalar, CpuCarrierEmitter carriers,
                    CpuLoopEmitter loops, CpuReductionEmitter reductions) { scalar.code().return_(); }
            @Override public void emitVector(CpuVectorEmitter vector, CpuCarrierEmitter carriers,
                    CpuLoopEmitter loops, CpuReductionEmitter reductions) { vector.code().return_(); }
        };
    }

    private static CpuFamilyKernelEmitter countingEmitter(AtomicInteger emissions) {
        return new CpuFamilyKernelEmitter() {
            @Override public CpuLoweringFingerprint loweringFingerprint() { return LOWERING; }
            @Override public void emitScalar(CpuScalarEmitter scalar, CpuCarrierEmitter carriers,
                    CpuLoopEmitter loops, CpuReductionEmitter reductions) {
                emissions.incrementAndGet(); scalar.code().return_();
            }
            @Override public void emitVector(CpuVectorEmitter vector, CpuCarrierEmitter carriers,
                    CpuLoopEmitter loops, CpuReductionEmitter reductions) {
                emissions.incrementAndGet(); vector.code().return_();
            }
        };
    }

    private static RuntimeException candidateFailure(CpuKernelSpecialization specialization,
            CpuFamilyKernelEmitter familyEmitter,
            List<PreparationResourceRequirement> requirements,
            List<CpuPortableKernelCandidate.BufferUse> bufferUses,
            List<CpuPortableKernelCandidate.WorkspaceUse> workspaceUses,
            CpuPortableInvocationBinder binder) {
        return assertThrows(RuntimeException.class, () -> new CpuPortableKernelCandidate(
                specialization, familyEmitter, requirements, bufferUses, workspaceUses, binder));
    }

    private static void assertExactSurface() {
        assertArrayEquals(new String[] {"workerCount", "minimumRangeSize", "deterministic"},
                Arrays.stream(CpuPreparedParallelConfiguration.class.getRecordComponents())
                        .map(component -> component.getName()).toArray(String[]::new));
        assertArrayEquals(new String[] {"supportedVectorShapes", "parallelConfiguration"},
                Arrays.stream(CpuPortableAnalysisInputs.class.getRecordComponents())
                        .map(component -> component.getName()).toArray(String[]::new));
        assertConstructor(CpuPortablePartitionPreparer.class, CpuPortableCandidateSource.class);
        assertConstructor(CpuPortablePartitionFinalizer.class, java.nio.file.Path.class,
                CpuWorkerGroup.class);
        assertConstructor(CpuPortablePreparationPlan.class, CpuPortablePartitionCandidate.class,
                CpuPreparedParallelConfiguration.class);
        assertConstructor(CpuPortableKernelCandidate.class, CpuKernelSpecialization.class,
                CpuFamilyKernelEmitter.class, List.class, List.class, List.class,
                CpuPortableInvocationBinder.class);
        assertEquals(Set.of("candidates"), publicContractMethods(CpuPortableCandidateSource.class));
        assertEquals(Set.of("bind"), publicContractMethods(CpuPortableInvocationBinder.class));
        assertEquals(Set.of("backendId", "finalizePartition"),
                publicContractMethods(CpuPortablePartitionFinalizer.class));
        assertEquals(Set.of("analyze"), publicContractMethods(CpuPortablePartitionPreparer.class));
    }

    private static Set<String> publicContractMethods(Class<?> type) {
        var names = new HashSet<String>();
        for (var method : type.getDeclaredMethods()) {
            if (!method.isSynthetic() && java.lang.reflect.Modifier.isPublic(method.getModifiers())) {
                names.add(method.getName());
            }
        }
        return names;
    }

    private static void assertConstructor(Class<?> type, Class<?>... parameterTypes) {
        var constructor = assertDoesNotThrow(() -> type.getDeclaredConstructor(parameterTypes));
        assertFalse(java.lang.reflect.Modifier.isPublic(constructor.getModifiers()), type.getName());
        assertEquals(1, type.getDeclaredConstructors().length, type.getName());
    }

    private enum SampleKind implements OperationKind {
        SAMPLE;
        @Override public List<OperationSignature> signatures() {
            return List.of(new OperationSignature(
                    NoOperationAttrs.class, 1, 1, 1, 1));
        }
    }

    private static final class ModifierAssertions {
        static boolean packagePrivate(Class<?> type) {
            return !java.lang.reflect.Modifier.isPublic(type.getModifiers())
                    && !java.lang.reflect.Modifier.isProtected(type.getModifiers())
                    && !java.lang.reflect.Modifier.isPrivate(type.getModifiers());
        }
        static boolean packagePrivateFinal(Class<?> type) {
            return packagePrivate(type) && java.lang.reflect.Modifier.isFinal(type.getModifiers());
        }
    }
}
