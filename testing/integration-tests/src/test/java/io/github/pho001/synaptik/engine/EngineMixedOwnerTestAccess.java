package io.github.pho001.synaptik.engine;

import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.runtime.run.RunState;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Test-only access to focused Engine ownership, lifecycle, and physical-identity evidence. */
public final class EngineMixedOwnerTestAccess {
    private EngineMixedOwnerTestAccess() {}

    /** Makes any later registry owner lookup fail without closing captured backend adapters. */
    public static void poisonBackendLookup(Engine engine) {
        Objects.requireNonNull(engine, "engine").poisonBackendLookupForTesting();
    }

    /** Returns exact planned owner names in partition order for one compiled graph. */
    public static List<String> partitionOwners(CompiledGraph graph) {
        Objects.requireNonNull(graph, "graph");
        Object artifacts = field(graph, "artifacts");
        List<?> partitions = (List<?>) invoke(artifacts, "partitions");
        var owners = new ArrayList<String>(partitions.size());
        for (Object partition : partitions) {
            Object owner = invoke(partition, "owner");
            owners.add((String) invoke(owner, "value"));
        }
        return List.copyOf(owners);
    }

    /** Returns exact planned owner names in partition order for one advanced compiled graph. */
    public static List<String> partitionOwners(AdvancedCompiledGraph graph) {
        Objects.requireNonNull(graph, "graph");
        Object artifacts = field(graph, "artifacts");
        List<?> partitions = (List<?>) invoke(artifacts, "partitions");
        var owners = new ArrayList<String>(partitions.size());
        for (Object partition : partitions) {
            Object owner = invoke(partition, "owner");
            owners.add((String) invoke(owner, "value"));
        }
        return List.copyOf(owners);
    }

    /** Returns the exact immutable Compiler artifacts retained by one compiled graph. */
    public static CompileArtifacts compileArtifacts(CompiledGraph graph) {
        Objects.requireNonNull(graph, "graph");
        return (CompileArtifacts) field(graph, "artifacts");
    }

    /** Returns the advanced owner behind an ordinary Engine for staged-gradient integration tests. */
    public static AdvancedEngine advancedEngine(Engine engine) {
        return Objects.requireNonNull(engine, "engine").lifecycleOwner();
    }

    /** Returns the exact immutable Compiler artifacts retained by one advanced compiled graph. */
    public static CompileArtifacts compileArtifacts(AdvancedCompiledGraph graph) {
        return Objects.requireNonNull(graph, "graph").artifacts();
    }

    /**
     * Returns identity-bearing inward publications and workspaces while the result remains open.
     */
    public static RunOwnedIdentities runOwnedIdentities(RunResult result) {
        Objects.requireNonNull(result, "result");
        AdvancedRunResult advanced = (AdvancedRunResult) field(result, "owner");
        io.github.pho001.synaptik.runtime.run.RunResult runtime =
                (io.github.pho001.synaptik.runtime.run.RunResult) field(advanced, "delegate");
        RunState state = (RunState) field(runtime, "runState");
        var publications = new ArrayList<Object>(runtime.resultCount());
        for (int index = 0; index < runtime.resultCount(); index++) {
            publications.add(runtime.publicationRepresentation(index));
        }
        var workspaces = new ArrayList<Object>(state.workspaceSlotCount());
        for (int index = 0; index < state.workspaceSlotCount(); index++) {
            workspaces.add(state.workspaceRepresentation(index));
        }
        return new RunOwnedIdentities(publications, workspaces);
    }

    private static Object field(Object owner, String name) {
        try {
            Field field = owner.getClass().getDeclaredField(name);
            field.setAccessible(true);
            return field.get(owner);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(
                    "cannot inspect " + owner.getClass().getName() + "." + name,
                    failure);
        }
    }

    private static Object invoke(Object owner, String name) {
        try {
            return owner.getClass().getMethod(name).invoke(owner);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(
                    "cannot invoke " + owner.getClass().getName() + "." + name,
                    failure);
        }
    }

    /** Identity-bearing snapshots; list elements are never copied or closed by this value. */
    public record RunOwnedIdentities(List<Object> publications, List<Object> workspaces) {
        /** Defensively snapshots non-null identity lists. */
        public RunOwnedIdentities {
            publications = List.copyOf(publications);
            workspaces = List.copyOf(workspaces);
        }
    }
}
