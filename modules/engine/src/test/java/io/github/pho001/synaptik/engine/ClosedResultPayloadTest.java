package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.memory.WorkspaceSlot;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import io.github.pho001.synaptik.runtime.run.RunResourceOwnership;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.io.File;
import java.lang.ref.Reference;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Bounded heap pressure checks owner reference release, not native release or a GC deadline. */
final class ClosedResultPayloadTest {
    private static final int PAYLOAD_BYTES = 1 << 20;
    private static final int RUNS = 128;
    @TempDir Path directory;

    @Test
    void retainedClosedOwnersReleasePayloadsEvenWhenEveryCleanupLayerFails() throws Exception {
        var classpath = new LinkedHashSet<String>();
        classpath.add(System.getProperty("java.class.path"));
        for (ClassLoader loader = getClass().getClassLoader(); loader != null; loader = loader.getParent()) {
            if (loader instanceof URLClassLoader urls) {
                for (var url : urls.getURLs()) classpath.add(Path.of(url.toURI()).toString());
            }
        }
        Path output = directory.resolve("closed-cleanup-failure.log");
        var builder = new ProcessBuilder(Path.of(System.getProperty("java.home"), "bin", "java").toString(),
                "-Xmx96m", "-cp", String.join(File.pathSeparator, classpath), getClass().getName())
                .redirectErrorStream(true).redirectOutput(output.toFile());
        builder.environment().remove("JDK_JAVA_OPTIONS");
        builder.environment().remove("JAVA_TOOL_OPTIONS");
        builder.environment().remove("_JAVA_OPTIONS");
        var process = builder.start();
        boolean finished = process.waitFor(90, TimeUnit.SECONDS);
        if (!finished) {
            process.destroyForcibly();
            process.waitFor();
        }
        String evidence = Files.readString(output);
        assertTrue(finished, () -> "Cleanup-failure child timed out:\n" + evidence);
        assertEquals(0, process.exitValue(), () -> "Cleanup-failure child failed:\n" + evidence);
        assertTrue(evidence.contains("retained_closed_failure_owners=128 PASS"), () -> evidence);
    }

    /**
     * Retains closed owners while each iteration allocates more payload than the heap can retain.
     * No collector request, reference queue, private-field inspection or collection deadline is used.
     *
     * @param args unused child-process arguments
     */
    public static void main(String[] args) {
        var retained = new ArrayList<AdvancedRunResult>();
        try (Engine engine = Engine.standard()) {
            for (int index = 0; index < RUNS; index++) {
                retained.add(closeFailingRun(engine.lifecycleOwner()));
            }
            for (AdvancedRunResult result : retained) {
                assertTrue(result.isClosed());
                assertEquals(2, result.resultCount());
            }
            System.out.println("retained_closed_failure_owners=" + retained.size() + " PASS");
            Reference.reachabilityFence(retained);
        }
    }

    private static AdvancedRunResult closeFailingRun(AdvancedEngine engine) {
        List<String> order = new ArrayList<>();
        Error primary = new AssertionError("workspace cleanup");
        RuntimeException ownedFailure = new RuntimeException("owned cleanup");
        RuntimeException wrapperFailure = new RuntimeException("wrapper cleanup");
        HeapBuffer caller = new HeapBuffer("caller", order, null);
        HeapBuffer owned = new HeapBuffer("owned", order, ownedFailure);
        HeapBuffer wrapper0 = new HeapBuffer("wrapper0", order, null);
        HeapBuffer wrapper1 = new HeapBuffer("wrapper1", order, wrapperFailure);
        var workspace = new HeapWorkspace(order, primary);
        var plan = new PreparedMemoryPlan(
                List.of(new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), PAYLOAD_BYTES, 1)),
                List.of(new PreparedMemoryPlan.WorkspaceEntry(new WorkspaceSlot(0), PAYLOAD_BYTES, 1)));
        var state = new RunState(plan, List.of(List.of(
                new BufferRepresentationBinding(caller, RunResourceOwnership.BORROWED),
                new BufferRepresentationBinding(owned, RunResourceOwnership.RUN_OWNED))), List.of(workspace));
        var geometry = state.geometry();
        var publication = new PreparedPublication(plan, 0, 0, 0).bind(state);
        var alias = new PreparedPublication(plan, 0, 0, 1).bind(state);
        publication.publish();
        alias.publish();
        var inward = new io.github.pho001.synaptik.runtime.run.RunResult(state, List.of(publication, alias));
        assertSame(caller, inward.publicationRepresentation(0));
        assertSame(caller, inward.publicationRepresentation(1));
        var outward = new AdvancedRunResult(engine, inward, List.of(wrapper0, wrapper1), List.of());

        assertSame(primary, assertThrows(AssertionError.class, outward::close));
        assertArrayEquals(new Throwable[] {ownedFailure, wrapperFailure}, primary.getSuppressed());
        assertEquals(List.of("workspace", "owned", "wrapper1", "wrapper0"), order);
        assertSame(primary, assertThrows(AssertionError.class, outward::close));
        assertDoesNotThrow(inward::close);
        assertDoesNotThrow(state::close);
        assertEquals(List.of("workspace", "owned", "wrapper1", "wrapper0"), order);
        assertEquals(1, owned.closeCount);
        assertEquals(1, workspace.closeCount);
        assertEquals(1, wrapper0.closeCount);
        assertEquals(1, wrapper1.closeCount);
        assertEquals(0, caller.closeCount);
        assertEquals(7, caller.payload[0]);
        assertSame(plan, state.memoryPlan());
        assertSame(geometry, state.geometry());
        assertEquals(1, state.bufferSlotCount());
        assertEquals(2, state.bufferRepresentationCount(0));
        assertEquals(1, state.workspaceSlotCount());
        assertEquals(2, inward.resultCount());
        assertEquals("run state is closed", assertThrows(IllegalStateException.class,
                () -> inward.publicationRepresentation(-1)).getMessage());
        assertEquals("run state is closed", assertThrows(IllegalStateException.class,
                () -> state.bufferRepresentation(-1, -1)).getMessage());
        assertEquals("run state is closed", assertThrows(IllegalStateException.class,
                () -> state.workspaceRepresentation(-1)).getMessage());
        return outward;
    }

    /** Deliberately keeps heap payload after close so the owner must release the representation. */
    private static final class HeapBuffer implements BufferRepresentation {
        private final byte[] payload = new byte[PAYLOAD_BYTES];
        private final String name;
        private final List<String> order;
        private final RuntimeException failure;
        private int closeCount;

        private HeapBuffer(String name, List<String> order, RuntimeException failure) {
            this.name = name;
            this.order = order;
            this.failure = failure;
            payload[0] = 7;
        }

        @Override public void close() {
            closeCount++;
            order.add(name);
            if (failure != null) throw failure;
        }
    }

    /** Deliberately retains its payload when physical cleanup reports failure. */
    private static final class HeapWorkspace implements WorkspaceRepresentation {
        private final byte[] payload = new byte[PAYLOAD_BYTES];
        private final List<String> order;
        private final Error failure;
        private int closeCount;

        private HeapWorkspace(List<String> order, Error failure) {
            this.order = order;
            this.failure = failure;
            payload[0] = 9;
        }

        @Override public void close() {
            closeCount++;
            order.add("workspace");
            throw failure;
        }
    }
}
