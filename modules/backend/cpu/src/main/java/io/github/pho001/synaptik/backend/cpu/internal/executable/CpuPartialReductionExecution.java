package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPartialReductionIr;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/** Executes a cold-bound generated modular partial recipe against run-owned workspace.
 *
 * <p>Workers invoke the generated partial body; after their one join, the caller invokes the
 * generated ordinal combine. The assigned workspace segment is used directly and is never copied
 * into a Java array.</p>
 */
final class CpuPartialReductionExecution {
    private CpuPartialReductionExecution() { }

    /**
     * Runs one INT32 recipe with one complete worker phase followed by the caller combine.
     *
     * @param artifact non-null cold-bound INT32 generated artifact
     * @param input non-null dense source array
     * @param inputBase non-negative first source element
     * @param output non-null dense destination array, mutated only after a successful worker join
     * @param outputBase non-negative first destination element
     * @param state non-null aligned run-owned state workspace with the recipe's exact byte size
     * @param workers worker group for multiple partial calls; non-null when required by the recipe
     */
    static void executeInt(CpuGeneratedKernel.PartialReductionArtifact artifact, int[] input,
            int inputBase, int[] output, int outputBase, MemorySegment state,
            CpuWorkerGroup workers) {
        execute(artifact, input, inputBase, output, outputBase, state, workers);
    }

    /**
     * Runs one INT64 recipe with one complete worker phase followed by the caller combine.
     *
     * @param artifact non-null cold-bound INT64 generated artifact
     * @param input non-null dense source array
     * @param inputBase non-negative first source element
     * @param output non-null dense destination array, mutated only after a successful worker join
     * @param outputBase non-negative first destination element
     * @param state non-null aligned run-owned state workspace with the recipe's exact byte size
     * @param workers worker group for multiple partial calls; non-null when required by the recipe
     */
    static void executeLong(CpuGeneratedKernel.PartialReductionArtifact artifact, long[] input,
            int inputBase, long[] output, int outputBase, MemorySegment state,
            CpuWorkerGroup workers) {
        execute(artifact, input, inputBase, output, outputBase, state, workers);
    }

    private static void execute(CpuGeneratedKernel.PartialReductionArtifact artifact,
            Object input, int inputBase, Object output, int outputBase, MemorySegment state,
            CpuWorkerGroup workers) {
        Objects.requireNonNull(artifact, "artifact"); Objects.requireNonNull(input, "input");
        Objects.requireNonNull(output, "output"); Objects.requireNonNull(state, "state");
        CpuPartialReductionIr ir = artifact.ir();
        int inputLength = input instanceof int[] values ? values.length : ((long[]) input).length;
        int outputLength = output instanceof int[] values ? values.length : ((long[]) output).length;
        if (inputBase < 0 || outputBase < 0 || state.byteSize() != ir.workspaceBytes()
                || state.address() % CpuPartialReductionIr.STATE_SLICE_BYTES != 0
                || Math.addExact(inputBase, Math.multiplyExact(ir.outputCount(), ir.domainCount())) > inputLength
                || Math.addExact(outputBase, ir.outputCount()) > outputLength) {
            throw new IllegalArgumentException("partial reduction invocation geometry disagrees");
        }
        CpuWorkerGroup.RangeCall[] calls = new CpuWorkerGroup.RangeCall[Math.toIntExact(
                Math.multiplyExact(ir.outputCount(), ir.partialCount()))];
        for (int cell = 0; cell < ir.outputCount(); cell++) for (int partial = 0;
                partial < ir.partialCount(); partial++) {
            int c = cell, p = partial;
            int begin = Math.toIntExact(Math.addExact(inputBase, Math.addExact(
                    Math.multiplyExact((long) c, ir.domainCount()), ir.begin(c, p))));
            int end = Math.toIntExact(Math.addExact(inputBase, Math.addExact(
                    Math.multiplyExact((long) c, ir.domainCount()), ir.end(c, p))));
            long offset = ir.stateOffset(c, p);
            calls[c * ir.partialCount() + p] = () -> invokePartial(artifact, input, begin, end, state, offset);
        }
        if (calls.length == 1) try { calls[0].invoke(); } catch (Throwable failure) {
            throw failed(failure);
        } else {
            if (workers == null) throw new IllegalArgumentException("partial route needs workers");
            workers.execute(calls);
        }
        try {
            if (input instanceof int[]) artifact.orderedCombine().invokeExact(state, 0,
                    Math.toIntExact(ir.outputCount()), (int[]) output, outputBase);
            else artifact.orderedCombine().invokeExact(state, 0, Math.toIntExact(ir.outputCount()),
                    (long[]) output, outputBase);
        } catch (Throwable failure) { throw failed(failure); }
    }

    private static void invokePartial(CpuGeneratedKernel.PartialReductionArtifact artifact,
            Object input, int begin, int end, MemorySegment state, long offset) throws Throwable {
        if (input instanceof int[] values) artifact.partialBody().invokeExact(values, begin, end, state, offset);
        else artifact.partialBody().invokeExact((long[]) input, begin, end, state, offset);
    }

    private static IllegalStateException failed(Throwable failure) {
        return new IllegalStateException("generated partial reduction failed", failure);
    }
}
