package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Exercises the shared production enqueue/wait algorithm without requiring a native library. */
final class CpuOneDnnWorkspaceFailureTest {
    @Test
    void partialEnqueueFailureStillWaitsAndKeepsOriginalWithWaitSuppressed() {
        var execute = new IllegalStateException("execute");
        var wait = new AssertionError("wait");
        try (var workspace = new FaultWorkspace(execute, wait)) {
            assertSame(execute, assertThrows(IllegalStateException.class, workspace::execute));
            assertEquals(List.of("enqueue", "wait"), workspace.calls);
            assertArrayEquals(new Throwable[] {wait}, execute.getSuppressed());
        }
    }

    @Test
    void executeErrorRemainsPrimaryAndIdenticalWaitFailureIsNotSelfSuppressed() {
        var failure = new AssertionError("execute and wait");
        try (var workspace = new FaultWorkspace(failure, failure)) {
            assertSame(failure, assertThrows(AssertionError.class, workspace::execute));
            assertEquals(List.of("enqueue", "wait"), workspace.calls);
            assertEquals(0, failure.getSuppressed().length);
        }
    }

    @Test
    void waitOnlyFailureAndExecuteOnlyFailureBothPropagateUnchanged() {
        var wait = new IllegalStateException("wait");
        try (var workspace = new FaultWorkspace(null, wait)) {
            assertSame(wait, assertThrows(IllegalStateException.class, workspace::execute));
            assertEquals(List.of("enqueue", "wait"), workspace.calls);
        }
        var execute = new AssertionError("execute");
        try (var workspace = new FaultWorkspace(execute, null)) {
            assertSame(execute, assertThrows(AssertionError.class, workspace::execute));
            assertEquals(List.of("enqueue", "wait"), workspace.calls);
            assertEquals(0, execute.getSuppressed().length);
        }
    }

    private static final class FaultWorkspace extends CpuOneDnnWorkspace {
        final List<String> calls = new ArrayList<>();
        private final Throwable executeFailure;
        private final Throwable waitFailure;
        FaultWorkspace(Throwable executeFailure, Throwable waitFailure) {
            super(new TestKernel());
            this.executeFailure = executeFailure;
            this.waitFailure = waitFailure;
        }
        @Override public void execute() { executeAndAwait(); }
        @Override void enqueue() { calls.add("enqueue"); failIfPresent(executeFailure); }
        @Override void await() { calls.add("wait"); failIfPresent(waitFailure); }
        @Override public void bind(MemorySegment[] inputs, MemorySegment output) {
            validateBindings(inputs, output);
        }
        @Override protected void closeNative() { calls.add("close"); }
        private static void failIfPresent(Throwable failure) {
            if (failure instanceof RuntimeException runtime) throw runtime;
            if (failure instanceof Error error) throw error;
        }
    }

    private static final class TestKernel implements CpuOneDnnKernel {
        private final CpuOneDnnKernelSpec.Tensor tensor = new CpuOneDnnKernelSpec.Tensor(
                DataType.FLOAT32, List.of(1L), List.of(1L));
        @Override public CpuOneDnnKernelSpec spec() {
            return new CpuOneDnnKernelSpec.Unary(CpuOneDnnKernelSpec.UnaryAlgorithm.ABS, tensor, tensor);
        }
        @Override public long scratchpadBytes() { return 0; }
        @Override public String implementation() { return "test control failure"; }
        @Override public CpuOneDnnWorkspace createWorkspace() { return new FaultWorkspace(null, null); }
        @Override public void close() { }
    }
}
