package io.github.pho001.synaptik.backend.cpu.execution;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable.*;
import io.github.pho001.synaptik.runtime.memory.*;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.run.*;
import java.io.IOException;
import java.lang.classfile.ClassFile;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.reflect.AccessFlag;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class CpuPortablePreparedExecutableTest {
    @Test
    void constructorValidatesPortableDependenciesAndBorrowedWorkerStateInOrder() {
        var specialization = CpuPortablePartitionPreparerTest.specialization(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                List.of(CpuPortablePartitionPreparerTest.argument(
                        CpuKernelSpecialization.Carrier.FLOAT_ARRAY)));
        var kernel = new CpuClassFileKernelGenerator().generate(
                specialization, CpuPortablePartitionPreparerTest.emitter());
        var plan = plan(16, 4);
        var selection = List.of(new BufferSelection(0, 0));
        var access = List.of(BufferAccess.READ_ONLY);
        var types = List.of(DataType.FLOAT32);
        var configuration = new CpuPreparedParallelConfiguration(1, 1, true);
        var binder = (CpuPortableInvocationBinder)
                (state, handle, spec, parallel, workers, buffers, workspaces) -> null;
        try (var workers = new CpuWorkerGroup(1)) {
            assertEquals("generatedKernel", assertThrows(NullPointerException.class,
                    () -> new CpuPortablePreparedExecutable(plan, selection, List.of(), access,
                            types, null, configuration, workers, binder)).getMessage());
            assertEquals("parallelConfiguration", assertThrows(NullPointerException.class,
                    () -> new CpuPortablePreparedExecutable(plan, selection, List.of(), access,
                            types, kernel, null, workers, binder)).getMessage());
            assertEquals("workerGroup", assertThrows(NullPointerException.class,
                    () -> new CpuPortablePreparedExecutable(plan, selection, List.of(), access,
                            types, kernel, configuration, null, binder)).getMessage());
            assertEquals("invocationBinder", assertThrows(NullPointerException.class,
                    () -> new CpuPortablePreparedExecutable(plan, selection, List.of(), access,
                            types, kernel, configuration, workers, null)).getMessage());
        }
        var mismatch = new CpuWorkerGroup(2);
        try {
            assertEquals("worker group count does not match prepared parallel configuration",
                    assertThrows(IllegalArgumentException.class,
                            () -> new CpuPortablePreparedExecutable(plan, selection, List.of(), access,
                                    types, kernel, configuration, mismatch, binder)).getMessage());
        } finally {
            mismatch.close();
        }
        var closed = new CpuWorkerGroup(1);
        closed.close();
        assertEquals("CPU worker group is closed", assertThrows(IllegalStateException.class,
                () -> new CpuPortablePreparedExecutable(plan, selection, List.of(), access,
                        types, kernel, configuration, closed, binder)).getMessage());
    }

    @Test
    void coldBindsAndDirectlyExecutesAllFourPortableModes() {
        for (CpuPortableExecutionMode mode : CpuPortableExecutionMode.values()) {
            var specialization = CpuPortablePartitionPreparerTest.specialization(mode,
                    List.of(CpuPortablePartitionPreparerTest.argument(
                            CpuKernelSpecialization.Carrier.FLOAT_ARRAY)));
            var kernel = new CpuClassFileKernelGenerator().generate(
                    specialization, CpuPortablePartitionPreparerTest.emitter());
            var memoryPlan = new PreparedMemoryPlan(
                    List.of(new PreparedMemoryPlan.BufferEntry(new BufferSlot(1), 16, 4)), List.of());
            var buffer = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(
                    DataType.FLOAT32, 4, MemorySegment.ofArray(new float[4])));
            var state = new RunState(memoryPlan,
                    List.of(List.of(borrowed(buffer))), List.of());
            try (var workers = new CpuWorkerGroup(2)) {
                var executable = new CpuPortablePreparedExecutable(memoryPlan,
                        List.of(new BufferSelection(0, 0)), List.of(),
                        List.of(BufferAccess.READ_ONLY), List.of(DataType.FLOAT32), kernel,
                        new CpuPreparedParallelConfiguration(2, 1, true), workers,
                        (runState, handle, ignoredSpecialization, parallel, workerGroup,
                                arguments, workspaces) -> {
                            var floats = assertInstanceOf(
                                    CpuBufferArgument.Floats.class, arguments[0]);
                            return mode.parallel()
                                    ? new ParallelInvocation(runState, handle, floats.carrier(),
                                            workerGroup, parallel)
                                    : new ScalarInvocation(runState, handle, floats.carrier());
                        });
                BoundInvocation invocation = executable.bind(state);
                invocation.execute();
                if (invocation instanceof ParallelInvocation parallel) {
                    assertEquals(2, parallel.rangeCalls.get());
                } else {
                    assertEquals(1, ((ScalarInvocation) invocation).calls);
                }
                assertSame(kernel, executable.generatedKernel());
                assertSame(kernel.entryPoint(), executable.entryPoint());
            } finally {
                state.close();
            }
        }
    }

    @Test
    void retainsMixedDirectFieldsAndRejectsWrongCarrierBeforeBinder() {
        var arguments = List.of(
                CpuPortablePartitionPreparerTest.argument(
                        CpuKernelSpecialization.Carrier.MEMORY_SEGMENT),
                CpuPortablePartitionPreparerTest.argument(
                        CpuKernelSpecialization.Carrier.FLOAT_ARRAY));
        var specialization = CpuPortablePartitionPreparerTest.specialization(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD, arguments);
        var kernel = new CpuClassFileKernelGenerator().generate(
                specialization, CpuPortablePartitionPreparerTest.emitter());
        var plan = new PreparedMemoryPlan(List.of(
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(1), 16, 4),
                new PreparedMemoryPlan.BufferEntry(new BufferSlot(2), 16, 4)), List.of());
        var nativeBuffer = CpuNativeBuffer.allocate(DataType.FLOAT32, 16, 4);
        var heap = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(
                DataType.FLOAT32, 4, MemorySegment.ofArray(new float[4])));
        var state = new RunState(plan, List.of(
                List.of(owned(nativeBuffer)), List.of(borrowed(heap))), List.of());
        try (var workers = new CpuWorkerGroup(1)) {
            var binderCalls = new int[1];
            var executable = new CpuPortablePreparedExecutable(plan,
                    List.of(new BufferSelection(0, 0), new BufferSelection(1, 0)), List.of(),
                    List.of(BufferAccess.READ_ONLY, BufferAccess.READ_ONLY),
                    List.of(DataType.FLOAT32, DataType.FLOAT32), kernel,
                    new CpuPreparedParallelConfiguration(1, 1, true), workers,
                    (runState, handle, ignoredSpecialization, parallel, workerGroup,
                            boundArguments, workspaces) -> {
                        binderCalls[0]++;
                        assertAll(() -> assertSame(kernel.entryPoint(), handle),
                                () -> assertSame(specialization, ignoredSpecialization),
                                () -> assertSame(workers, workerGroup));
                        var segment = assertInstanceOf(
                                CpuBufferArgument.Segment.class, boundArguments[0]);
                        var floats = assertInstanceOf(
                                CpuBufferArgument.Floats.class, boundArguments[1]);
                        return new MixedInvocation(runState, handle, segment.segment(), floats.carrier());
                    });
            var invocation = assertInstanceOf(MixedInvocation.class, executable.bind(state));
            invocation.execute();
            assertAll(
                    () -> assertEquals(1, binderCalls[0]),
                    () -> assertSame(nativeBuffer.segment(), invocation.segment),
                    () -> assertSame(kernel.entryPoint(), invocation.handle),
                    () -> assertEquals(1, invocation.calls),
                    () -> assertTrue(Arrays.stream(MixedInvocation.class.getDeclaredFields())
                            .noneMatch(field -> field.getType() == CpuBufferArgument[].class
                                    || field.getType() == CpuNativeWorkspace[].class
                                    || BufferRepresentation.class.isAssignableFrom(field.getType())
                                    || HostTensorStorage.class.isAssignableFrom(field.getType()))));
            var wrongHeap = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(
                    DataType.FLOAT32, 4, MemorySegment.ofArray(new float[4])));
            var wrongState = new RunState(plan, List.of(
                    List.of(borrowed(wrongHeap)), List.of(borrowed(heap))), List.of());
            try {
                assertEquals("bufferArguments[0] does not match specialization carrier",
                        assertThrows(IllegalArgumentException.class,
                                () -> executable.bind(wrongState)).getMessage());
                assertEquals(1, binderCalls[0]);
            } finally {
                wrongState.close();
            }
        } finally {
            state.close();
        }
    }

    @Test
    void parallelModeRejectsSegmentUnavailableToWorkersBeforeBinder() {
        var specialization = CpuPortablePartitionPreparerTest.specialization(
                CpuPortableExecutionMode.SCALAR_PARALLEL,
                List.of(CpuPortablePartitionPreparerTest.argument(
                        CpuKernelSpecialization.Carrier.MEMORY_SEGMENT)));
        var kernel = new CpuClassFileKernelGenerator().generate(
                specialization, CpuPortablePartitionPreparerTest.emitter());
        var plan = new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(new BufferSlot(1), 16, 4)), List.of());
        try (var arena = Arena.ofConfined(); var workers = new CpuWorkerGroup(1)) {
            var borrowed = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(
                    DataType.FLOAT32, 4, arena.allocate(16, 4)));
            var state = new RunState(plan, List.of(List.of(borrowed(borrowed))), List.of());
            try {
                var executable = new CpuPortablePreparedExecutable(plan,
                        List.of(new BufferSelection(0, 0)), List.of(),
                        List.of(BufferAccess.READ_ONLY), List.of(DataType.FLOAT32), kernel,
                        new CpuPreparedParallelConfiguration(1, 1, true), workers,
                        (runState, handle, spec, parallel, workerGroup, boundArguments, workspaces) ->
                                fail("binder must not be called"));
                assertEquals("bufferArguments[0] is not accessible by every CPU worker",
                        assertThrows(IllegalArgumentException.class,
                                () -> executable.bind(state)).getMessage());
            } finally {
                state.close();
            }
        }
    }

    @Test
    void immutableExecutableBindsRepeatedlyAndConcurrentlyToDistinctRunStates() throws Exception {
        var specialization = CpuPortablePartitionPreparerTest.specialization(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                List.of(CpuPortablePartitionPreparerTest.argument(
                        CpuKernelSpecialization.Carrier.FLOAT_ARRAY)));
        var kernel = new CpuClassFileKernelGenerator().generate(
                specialization, CpuPortablePartitionPreparerTest.emitter());
        var plan = new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(new BufferSlot(1), 16, 4)), List.of());
        try (var workers = new CpuWorkerGroup(2);
                var executor = Executors.newFixedThreadPool(2)) {
            var executable = new CpuPortablePreparedExecutable(plan,
                    List.of(new BufferSelection(0, 0)), List.of(),
                    List.of(BufferAccess.READ_ONLY), List.of(DataType.FLOAT32), kernel,
                    new CpuPreparedParallelConfiguration(2, 1, true), workers,
                    (state, handle, spec, parallel, group, buffers, workspaces) ->
                            new ScalarInvocation(state, handle,
                                    ((CpuBufferArgument.Floats) buffers[0]).carrier()));
            var first = state(plan);
            var second = state(plan);
            try {
                var firstFuture = executor.submit(() -> executable.bind(first));
                var secondFuture = executor.submit(() -> executable.bind(second));
                var firstInvocation = assertInstanceOf(ScalarInvocation.class, firstFuture.get());
                var secondInvocation = assertInstanceOf(ScalarInvocation.class, secondFuture.get());
                assertNotSame(firstInvocation, secondInvocation);
                firstInvocation.execute();
                secondInvocation.execute();
                first.close();
                assertEquals("run state is closed", assertThrows(IllegalStateException.class,
                        firstInvocation::execute).getMessage());
                assertDoesNotThrow(secondInvocation::execute);
            } finally {
                if (!first.isClosed()) first.close();
                if (!second.isClosed()) second.close();
            }
        }
    }

    @Test
    void portablePathAcceptsReadOnlyHeapAsExactSegmentAndRejectsWriteAccess() {
        var specialization = CpuPortablePartitionPreparerTest.specialization(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                List.of(CpuPortablePartitionPreparerTest.argument(
                        CpuKernelSpecialization.Carrier.MEMORY_SEGMENT)));
        var kernel = new CpuClassFileKernelGenerator().generate(
                specialization, CpuPortablePartitionPreparerTest.emitter());
        var plan = plan(16, 4);
        var exact = MemorySegment.ofArray(new float[4]).asReadOnly();
        var buffer = CpuBorrowedBuffer.borrow(
                new MemorySegmentStorage(DataType.FLOAT32, 4, exact));
        var state = new RunState(plan, List.of(List.of(borrowed(buffer))), List.of());
        try (var workers = new CpuWorkerGroup(1)) {
            var binderCalls = new int[1];
            var read = executable(plan, List.of(new BufferSelection(0, 0)), List.of(),
                    List.of(BufferAccess.READ_ONLY), kernel, workers,
                    (runState, handle, spec, parallel, group, buffers, workspaces) -> {
                        binderCalls[0]++;
                        var segment = assertInstanceOf(CpuBufferArgument.Segment.class, buffers[0]);
                        assertAll(() -> assertSame(exact, segment.segment()),
                                () -> assertTrue(segment.readOnly()),
                                () -> assertEquals(16, segment.byteSize()),
                                () -> assertEquals(0, segment.byteOffset()));
                        return new SegmentInvocation(runState, handle, segment.segment());
                    });
            read.bind(state).execute();
            assertEquals(1, binderCalls[0]);
            var write = executable(plan, List.of(new BufferSelection(0, 0)), List.of(),
                    List.of(BufferAccess.WRITE_ONLY), kernel, workers,
                    (runState, handle, spec, parallel, group, buffers, workspaces) ->
                            fail("binder must not be called"));
            assertIncompatible(write, state);
        } finally {
            state.close();
        }
    }

    @Test
    void validatesDynamicAndBakedArrayOffsetsBeforeOneTypedBinderCall() {
        var dynamicArgument = new CpuKernelSpecialization.Argument(DataType.FLOAT32,
                CpuKernelSpecialization.Carrier.FLOAT_ARRAY, BufferAccess.READ_ONLY,
                false, 0, List.of(1L));
        var dynamicSpecialization = CpuPortablePartitionPreparerTest.specialization(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD, List.of(dynamicArgument));
        var dynamicKernel = new CpuClassFileKernelGenerator().generate(
                dynamicSpecialization, CpuPortablePartitionPreparerTest.emitter());
        var plan = plan(16, 4);
        var array = new float[5];
        var slice = MemorySegment.ofArray(array).asSlice(Float.BYTES, 16);
        var buffer = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT32, 4, slice));
        var state = new RunState(plan, List.of(List.of(borrowed(buffer))), List.of());
        try (var workers = new CpuWorkerGroup(1)) {
            var calls = new int[1];
            var dynamic = executable(plan, List.of(new BufferSelection(0, 0)), List.of(),
                    List.of(BufferAccess.READ_ONLY), dynamicKernel, workers,
                    (runState, handle, spec, parallel, group, buffers, workspaces) -> {
                        calls[0]++;
                        var floats = assertInstanceOf(CpuBufferArgument.Floats.class, buffers[0]);
                        assertAll(() -> assertSame(array, floats.carrier()),
                                () -> assertEquals(Float.BYTES, floats.byteOffset()));
                        return new DynamicOffsetInvocation(runState, handle, floats.carrier(),
                                floats.byteOffset());
                    });
            dynamic.bind(state).execute();
            assertEquals(1, calls[0]);

            var bakedArgument = new CpuKernelSpecialization.Argument(DataType.FLOAT32,
                    CpuKernelSpecialization.Carrier.FLOAT_ARRAY, BufferAccess.READ_ONLY,
                    true, Float.BYTES, List.of(1L));
            var bakedSpec = CpuPortablePartitionPreparerTest.specialization(
                    CpuPortableExecutionMode.SCALAR_SINGLE_THREAD, List.of(bakedArgument));
            var bakedKernel = new CpuClassFileKernelGenerator().generate(
                    bakedSpec, CpuPortablePartitionPreparerTest.emitter());
            var baked = executable(plan, List.of(new BufferSelection(0, 0)), List.of(),
                    List.of(BufferAccess.READ_ONLY), bakedKernel, workers,
                    (runState, handle, spec, parallel, group, buffers, workspaces) ->
                            new ScalarInvocation(runState, handle,
                                    ((CpuBufferArgument.Floats) buffers[0]).carrier()));
            assertDoesNotThrow(() -> baked.bind(state));

            var zeroKernel = new CpuClassFileKernelGenerator().generate(
                    CpuPortablePartitionPreparerTest.specialization(
                            CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                            List.of(CpuPortablePartitionPreparerTest.argument(
                                    CpuKernelSpecialization.Carrier.FLOAT_ARRAY))),
                    CpuPortablePartitionPreparerTest.emitter());
            var mismatch = executable(plan, List.of(new BufferSelection(0, 0)), List.of(),
                    List.of(BufferAccess.READ_ONLY), zeroKernel, workers,
                    (runState, handle, spec, parallel, group, buffers, workspaces) ->
                            fail("binder must not be called"));
            assertEquals("bufferArguments[0] byte offset does not match specialization",
                    assertThrows(IllegalArgumentException.class,
                            () -> mismatch.bind(state)).getMessage());
        } finally {
            state.close();
        }
    }

    @Test
    void portablePathRetainsColdLivenessAlignmentAndThreadAccessChecks() throws Exception {
        var specialization = CpuPortablePartitionPreparerTest.specialization(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                List.of(CpuPortablePartitionPreparerTest.argument(
                        CpuKernelSpecialization.Carrier.MEMORY_SEGMENT)));
        var kernel = new CpuClassFileKernelGenerator().generate(
                specialization, CpuPortablePartitionPreparerTest.emitter());
        var plan = plan(4, 8);
        try (var workers = new CpuWorkerGroup(1)) {
            var executable = executable(plan, List.of(new BufferSelection(0, 0)), List.of(),
                    List.of(BufferAccess.READ_ONLY), kernel, workers,
                    (runState, handle, spec, parallel, group, buffers, workspaces) ->
                            fail("binder must not be called"));
            var deadArena = Arena.ofShared();
            var dead = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(
                    DataType.FLOAT32, 1, deadArena.allocate(4, 8)));
            var deadState = new RunState(plan, List.of(List.of(borrowed(dead))), List.of());
            deadArena.close();
            assertIncompatible(executable, deadState);
            deadState.close();
            try (var arena = Arena.ofShared()) {
                var segment = arena.allocate(12, 8).asSlice(4, 4);
                var misaligned = CpuBorrowedBuffer.borrow(
                        new MemorySegmentStorage(DataType.FLOAT32, 1, segment));
                var state = new RunState(plan, List.of(List.of(borrowed(misaligned))), List.of());
                assertIncompatible(executable, state);
                state.close();
            }
            try (var arena = Arena.ofConfined(); var executor = Executors.newSingleThreadExecutor()) {
                var confined = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(
                        DataType.FLOAT32, 1, arena.allocate(4, 8)));
                var state = new RunState(plan, List.of(List.of(borrowed(confined))), List.of());
                var failure = executor.submit(() -> assertThrows(IllegalArgumentException.class,
                        () -> executable.bind(state))).get();
                assertEquals("bufferSelections[0] is incompatible with prepared executable",
                        failure.getMessage());
                state.close();
            }
        }
    }

    @Test
    void parallelModeRejectsWorkspaceUnavailableToWorkersBeforeBinder() throws Exception {
        var specialization = CpuPortablePartitionPreparerTest.specialization(
                CpuPortableExecutionMode.SCALAR_PARALLEL, List.of());
        var kernel = new CpuClassFileKernelGenerator().generate(
                specialization, CpuPortablePartitionPreparerTest.emitter());
        var plan = new PreparedMemoryPlan(List.of(), List.of(
                new PreparedMemoryPlan.WorkspaceEntry(new WorkspaceSlot(1), 8, 8)));
        var arena = Arena.ofConfined();
        var constructor = CpuNativeWorkspace.class.getDeclaredConstructor(
                long.class, long.class, Arena.class, MemorySegment.class);
        constructor.setAccessible(true);
        var workspace = constructor.newInstance(8L, 8L, arena, arena.allocate(8, 8));
        var state = new RunState(plan, List.of(), List.of(workspace));
        try (var workers = new CpuWorkerGroup(1)) {
            var executable = executable(plan, List.of(), List.of(new WorkspaceSelection(0)),
                    List.of(), kernel, workers,
                    (runState, handle, spec, parallel, group, buffers, workspaces) ->
                            fail("binder must not be called"));
            assertEquals("workspaces[0] is not accessible by every CPU worker",
                    assertThrows(IllegalArgumentException.class,
                            () -> executable.bind(state)).getMessage());
        } finally {
            state.close();
        }
    }

    @Test
    void rejectsNullAndWrongStateBinderResultsAfterExactlyOneCall() {
        var specialization = CpuPortablePartitionPreparerTest.specialization(
                CpuPortableExecutionMode.SCALAR_SINGLE_THREAD,
                List.of(CpuPortablePartitionPreparerTest.argument(
                        CpuKernelSpecialization.Carrier.FLOAT_ARRAY)));
        var kernel = new CpuClassFileKernelGenerator().generate(
                specialization, CpuPortablePartitionPreparerTest.emitter());
        var plan = plan(16, 4);
        var state = state(plan);
        var other = state(plan);
        try (var workers = new CpuWorkerGroup(1)) {
            var nullCalls = new int[1];
            var nullBinder = executable(plan, List.of(new BufferSelection(0, 0)), List.of(),
                    List.of(BufferAccess.READ_ONLY), kernel, workers,
                    (runState, handle, spec, parallel, group, buffers, workspaces) -> {
                        nullCalls[0]++; return null;
                    });
            assertEquals("boundInvocation", assertThrows(NullPointerException.class,
                    () -> nullBinder.bind(state)).getMessage());
            assertEquals(1, nullCalls[0]);
            var wrongCalls = new int[1];
            var wrong = executable(plan, List.of(new BufferSelection(0, 0)), List.of(),
                    List.of(BufferAccess.READ_ONLY), kernel, workers,
                    (runState, handle, spec, parallel, group, buffers, workspaces) -> {
                        wrongCalls[0]++;
                        return new ScalarInvocation(other, handle,
                                ((CpuBufferArgument.Floats) buffers[0]).carrier());
                    });
            assertEquals("bound invocation does not belong to supplied run state",
                    assertThrows(IllegalArgumentException.class, () -> wrong.bind(state)).getMessage());
            assertEquals(1, wrongCalls[0]);
        } finally {
            state.close(); other.close();
        }
    }

    @Test
    void productionSourceAndClassFilesExcludeForbiddenPortableHotMechanisms() throws IOException {
        var types = List.of(CpuPreparedParallelConfiguration.class, CpuPortableAnalysisInputs.class,
                CpuPortableCandidateSource.class, CpuPortableKernelCandidate.class,
                CpuPortableInvocationBinder.class, CpuPortablePreparationPlan.class,
                CpuPortablePartitionPreparer.class, CpuPortablePartitionFinalizer.class,
                CpuPortablePreparedExecutable.class);
        var forbidden = List.of("HostTensorStorage", "invokeWithArguments", ".asType(",
                ".asSpreader(", ".asCollector(", ".bindTo(", "Object[]", "java.lang.reflect",
                "ServiceLoader", "service locator", "vendor",
                "operation capability", "Runtime discovery");
        for (Class<?> type : types) {
            assertFalse(java.lang.reflect.Modifier.isPublic(type.getModifiers()), type.getName());
            byte[] bytes;
            try (var stream = type.getResourceAsStream(type.getSimpleName() + ".class")) {
                assertNotNull(stream);
                bytes = stream.readAllBytes();
            }
            var model = ClassFile.of().parse(bytes);
            assertFalse(model.flags().has(AccessFlag.PUBLIC), type.getName());
            assertTrue(ClassFile.of().verify(bytes).isEmpty(), type.getName());
            String constants = new String(bytes, StandardCharsets.ISO_8859_1);
            for (String token : forbidden) assertFalse(constants.contains(token),
                    type.getName() + " class file contains " + token);
            String source = Files.readString(productionSource(type));
            for (String token : forbidden) assertFalse(source.contains(token),
                    type.getName() + " source contains " + token);
            assertAll(() -> assertFalse(source.contains("public class "), type.getName()),
                    () -> assertFalse(source.contains("public record "), type.getName()),
                    () -> assertFalse(source.contains("public interface "), type.getName()));
        }
        String source = Files.readString(productionSource(CpuPortablePreparedExecutable.class));
        assertAll(() -> assertFalse(source.contains("instanceof Object")),
                () -> assertFalse(source.contains("switch (bufferArguments")),
                () -> assertFalse(source.contains("getClass(")),
                () -> assertFalse(source.contains("Class.forName")),
                () -> assertFalse(source.contains("MethodHandles.lookup")));
        String analysisSource = Files.readString(productionSource(CpuPortableKernelCandidate.class))
                + Files.readString(productionSource(CpuPortablePartitionPreparer.class));
        for (String token : List.of("loadOrGenerate(", ".generate(", ".emitScalar(",
                ".emitVector(", "defineHiddenClass", "Files.", "CpuNativeBuffer.allocate(",
                "workerGroup.execute(")) {
            assertFalse(analysisSource.contains(token), "analysis source contains " + token);
        }
        String finalizerSource = Files.readString(productionSource(CpuPortablePartitionFinalizer.class));
        assertEquals(1, occurrences(finalizerSource, "loadOrGenerate("));
    }

    private static RunState state(PreparedMemoryPlan plan) {
        var buffer = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(
                DataType.FLOAT32, 4, MemorySegment.ofArray(new float[4])));
        return new RunState(plan, List.of(List.of(borrowed(buffer))), List.of());
    }

    private static PreparedMemoryPlan plan(long byteSize, long byteAlignment) {
        return new PreparedMemoryPlan(List.of(new PreparedMemoryPlan.BufferEntry(
                new BufferSlot(1), byteSize, byteAlignment)), List.of());
    }

    private static CpuPortablePreparedExecutable executable(PreparedMemoryPlan plan,
            List<BufferSelection> buffers, List<WorkspaceSelection> workspaces,
            List<BufferAccess> accesses, CpuGeneratedKernel kernel, CpuWorkerGroup workers,
            CpuPortableInvocationBinder binder) {
        var types = new java.util.ArrayList<DataType>(buffers.size());
        for (int index = 0; index < buffers.size(); index++) types.add(DataType.FLOAT32);
        return new CpuPortablePreparedExecutable(plan, buffers, workspaces, accesses, types,
                kernel, new CpuPreparedParallelConfiguration(workers.workerCount(), 1, true),
                workers, binder);
    }

    private static void assertIncompatible(CpuPortablePreparedExecutable executable,
            RunState state) {
        assertEquals("bufferSelections[0] is incompatible with prepared executable",
                assertThrows(IllegalArgumentException.class, () -> executable.bind(state)).getMessage());
    }

    private static Path productionSource(Class<?> type) {
        String relative = "src/main/java/" + type.getName().replace('.', '/') + ".java";
        Path moduleRelative = Path.of(relative);
        if (Files.exists(moduleRelative)) return moduleRelative;
        return Path.of("backends/cpu").resolve(relative);
    }

    private static int occurrences(String source, String token) {
        int count = 0;
        for (int index = source.indexOf(token); index >= 0;
                index = source.indexOf(token, index + token.length())) count++;
        return count;
    }

    private static BufferRepresentationBinding borrowed(CpuBufferRepresentation value) {
        return new BufferRepresentationBinding(value, RunResourceOwnership.BORROWED);
    }
    private static BufferRepresentationBinding owned(CpuBufferRepresentation value) {
        return new BufferRepresentationBinding(value, RunResourceOwnership.RUN_OWNED);
    }

    private static final class ScalarInvocation extends BoundInvocation
            implements CpuPortableKernelInvocation {
        private final MethodHandle handle;
        private final float[] carrier;
        int calls;
        ScalarInvocation(RunState state, MethodHandle handle, float[] carrier) {
            super(state); this.handle = handle; this.carrier = carrier;
        }
        @Override protected void executeBound() {
            try { handle.invokeExact(carrier, 4L); calls++; }
            catch (RuntimeException | Error failure) { throw failure; }
            catch (Throwable failure) { throw new AssertionError(failure); }
        }
    }

    private static final class ParallelInvocation extends BoundInvocation
            implements CpuPortableKernelInvocation {
        private final MethodHandle handle;
        private final float[] carrier;
        private final CpuWorkerGroup workers;
        private final CpuPreparedParallelConfiguration parallel;
        final AtomicInteger rangeCalls = new AtomicInteger();
        ParallelInvocation(RunState state, MethodHandle handle, float[] carrier,
                CpuWorkerGroup workers, CpuPreparedParallelConfiguration parallel) {
            super(state); this.handle = handle; this.carrier = carrier;
            this.workers = workers; this.parallel = parallel;
        }
        @Override protected void executeBound() {
            workers.execute(0, 4, parallel.minimumRangeSize(), parallel.deterministic(),
                    (start, end, index) -> invokeRange(start, end, index));
        }
        private void invokeRange(long start, long end, int index) {
            try { handle.invokeExact(carrier, start, end, index); rangeCalls.incrementAndGet(); }
            catch (RuntimeException | Error failure) { throw failure; }
            catch (Throwable failure) { throw new AssertionError(failure); }
        }
    }

    private static final class MixedInvocation extends BoundInvocation
            implements CpuPortableKernelInvocation {
        private final MethodHandle handle;
        final MemorySegment segment;
        private final float[] carrier;
        int calls;
        MixedInvocation(RunState state, MethodHandle handle, MemorySegment segment, float[] carrier) {
            super(state); this.handle = handle; this.segment = segment; this.carrier = carrier;
        }
        @Override protected void executeBound() {
            try { handle.invokeExact(segment, carrier, 4L); calls++; }
            catch (RuntimeException | Error failure) { throw failure; }
            catch (Throwable failure) { throw new AssertionError(failure); }
        }
    }

    private static final class SegmentInvocation extends BoundInvocation
            implements CpuPortableKernelInvocation {
        private final MethodHandle handle;
        private final MemorySegment segment;
        SegmentInvocation(RunState state, MethodHandle handle, MemorySegment segment) {
            super(state); this.handle = handle; this.segment = segment;
        }
        @Override protected void executeBound() {
            try { handle.invokeExact(segment, 4L); }
            catch (RuntimeException | Error failure) { throw failure; }
            catch (Throwable failure) { throw new AssertionError(failure); }
        }
    }

    private static final class DynamicOffsetInvocation extends BoundInvocation
            implements CpuPortableKernelInvocation {
        private final MethodHandle handle;
        private final float[] carrier;
        private final long byteOffset;
        DynamicOffsetInvocation(RunState state, MethodHandle handle, float[] carrier,
                long byteOffset) {
            super(state); this.handle = handle; this.carrier = carrier; this.byteOffset = byteOffset;
        }
        @Override protected void executeBound() {
            try { handle.invokeExact(carrier, byteOffset, 4L); }
            catch (RuntimeException | Error failure) { throw failure; }
            catch (Throwable failure) { throw new AssertionError(failure); }
        }
    }
}
