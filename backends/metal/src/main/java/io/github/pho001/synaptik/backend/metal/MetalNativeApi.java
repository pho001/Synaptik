package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import io.github.pho001.synaptik.config.compile.NumericalProfile;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.file.Path;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Package-private typed seam for the ABI-v4 Metal foundation, MPSGraph, and custom-NEG C ABI.
 *
 * <p>Handles remain opaque carrier segments inside this package. Implementations consume each
 * successful context or buffer handle exactly once through its matching release call. Native
 * status failures are unchecked and retain both the operation name and raw status value.</p>
 */
abstract class MetalNativeApi implements AutoCloseable {
    static final int ABI_VERSION = 4;
    static final String EXECUTABLE_CREATE_OPERATION =
            "synaptik_metal_mpsgraph_executable_create";
    static final String EXECUTABLE_RELEASE_OPERATION =
            "synaptik_metal_mpsgraph_executable_release";
    static final String EXECUTABLE_RUN_OPERATION =
            "synaptik_metal_mpsgraph_executable_run";
    static final String NEG_KERNEL_PIPELINE_CREATE_OPERATION =
            "synaptik_metal_neg_kernel_pipeline_create";
    static final String NEG_KERNEL_PIPELINE_RELEASE_OPERATION =
            "synaptik_metal_neg_kernel_pipeline_release";
    static final String NEG_KERNEL_PIPELINE_RUN_OPERATION =
            "synaptik_metal_neg_kernel_pipeline_run";

    /**
     * Opens and completely validates the production ABI without creating a Metal context.
     *
     * @param absoluteLibraryPath caller-selected absolute library path; not retained as mutable
     *     state and must not be {@code null}
     * @return a new open API owner whose symbol-lookup lifetime must be closed; never {@code null}
     * @throws NullPointerException if {@code absoluteLibraryPath} is {@code null}
     * @throws IllegalArgumentException if the path is not absolute
     * @throws RuntimeException if loading, symbol resolution, or ABI validation fails
     * @throws Error if native linking or failure cleanup reports an error
     */
    static MetalNativeApi open(Path absoluteLibraryPath) {
        Objects.requireNonNull(absoluteLibraryPath, "absoluteLibraryPath");
        if (!absoluteLibraryPath.isAbsolute()) {
            throw new IllegalArgumentException("absoluteLibraryPath must be absolute");
        }
        return Ffm.open(absoluteLibraryPath);
    }

    /**
     * Creates one native default-device and command-queue context.
     *
     * @return a fresh non-null opaque handle owned by the caller
     * @throws RuntimeException if native creation fails or violates the ABI output contract
     */
    abstract Handle createContext();

    /**
     * Consumes one live context handle exactly once.
     *
     * @param context non-null live handle owned by the caller
     * @throws RuntimeException if native release fails
     */
    abstract void releaseContext(Handle context);

    /**
     * Creates one fresh shared-storage buffer for the supplied logical byte size.
     *
     * @param context non-null live context handle; ownership remains with the caller
     * @param logicalByteSize exact non-negative logical byte extent
     * @return a fresh non-null opaque buffer handle owned by the caller
     * @throws IllegalArgumentException if {@code logicalByteSize} is negative
     * @throws RuntimeException if native allocation fails or violates the ABI output contract
     */
    abstract Handle createBuffer(Handle context, long logicalByteSize);

    /**
     * Consumes one live buffer handle exactly once.
     *
     * @param buffer non-null live handle owned by the caller
     * @throws RuntimeException if native release fails
     */
    abstract void releaseBuffer(Handle buffer);

    /**
     * Copies one already validated byte range from caller-owned memory into a live buffer.
     *
     * @param buffer non-null live buffer handle; ownership remains with the caller
     * @param bufferOffset non-negative logical destination offset
     * @param source non-null native source slice, or the null-address segment for a zero-byte copy
     * @param byteCount non-negative number of bytes to copy
     * @throws RuntimeException if the native copy fails
     */
    abstract void upload(Handle buffer, long bufferOffset, MemorySegment source, long byteCount);

    /**
     * Copies one already validated byte range from a live buffer into caller-owned memory.
     *
     * @param buffer non-null live buffer handle; ownership remains with the caller
     * @param bufferOffset non-negative logical source offset
     * @param destination non-null writable native destination slice, or the null-address segment
     *     for a zero-byte copy
     * @param byteCount non-negative number of bytes to copy
     * @throws RuntimeException if the native copy fails
     */
    abstract void download(
            Handle buffer, long bufferOffset, MemorySegment destination, long byteCount);

    /**
     * Compiles one shape-specialized whole-partition typed MPSGraph executable.
     *
     * @param context non-null live context whose ownership remains with the caller
     * @param numericalProfile non-null cold plan profile used by Java fail-closed preflight
     * @param valueRanks non-null value-aligned ranks
     * @param valueDimensions non-null row-major value-count by sixteen dimension table
     * @param graphProgram non-null version-eight typed node table
     * @param feedValueIndices non-null stable feed value indices
     * @param targetValueIndices non-null stable target value indices
     * @return a fresh non-null opaque executable handle owned by the caller
     * @throws RuntimeException if Java validation, native construction, validation, or compilation
     *     fails
     */
    final Handle createMpsGraphExecutable(
            Handle context,
            NumericalProfile numericalProfile,
            int[] valueRanks,
            long[] valueDimensions,
            MetalMpsGraphProgram graphProgram,
            int[] feedValueIndices,
            int[] targetValueIndices) {
        Objects.requireNonNull(context, "context");
        MpsGraphExecutableAbi.validateCreate(
                numericalProfile,
                valueRanks, valueDimensions, graphProgram,
                feedValueIndices, targetValueIndices);
        NativeCreateResult result = Objects.requireNonNull(createMpsGraphExecutableNative(
                context, valueRanks, valueDimensions, graphProgram,
                feedValueIndices, targetValueIndices),
                "native executable create result");
        return finishExecutableCreate(result);
    }

    /**
     * Performs the already validated native create invocation.
     *
     * @param context non-null live context whose ownership remains with the caller
     * @param valueRanks validated value-aligned ranks
     * @param valueDimensions validated padded dimension table
     * @param graphProgram validated version-eight typed topological node table
     * @param feedValueIndices validated unique feeds
     * @param targetValueIndices validated unique produced targets
     * @return non-null raw status/output-cell result for checked interpretation
     * @throws RuntimeException if the native invocation itself fails
     */
    abstract NativeCreateResult createMpsGraphExecutableNative(
            Handle context,
            int[] valueRanks,
            long[] valueDimensions,
            MetalMpsGraphProgram graphProgram,
            int[] feedValueIndices,
            int[] targetValueIndices);

    /**
     * Consumes one live native executable handle exactly once.
     *
     * @param executable non-null live handle owned by the caller
     * @throws RuntimeException if native release fails
     */
    final void releaseExecutable(Handle executable) {
        Objects.requireNonNull(executable, "executable");
        checkExecutableStatus(EXECUTABLE_RELEASE_OPERATION,
                releaseExecutableNative(executable));
    }

    /**
     * Returns the raw status from one already validated executable release invocation.
     *
     * @param executable non-null live executable handle to consume
     * @return exact raw native status
     * @throws RuntimeException if the native invocation itself fails
     */
    abstract int releaseExecutableNative(Handle executable);

    /**
     * Executes one compiled region with stable ordered direct buffer bindings.
     *
     * @param executable non-null live executable whose ownership remains with the caller
     * @param inputBuffers non-null stable feed-ordered live buffer handles
     * @param outputBuffers non-null stable target-ordered live buffer handles
     * @throws RuntimeException if validation or synchronous execution fails
     */
    final void runExecutable(
            Handle executable, int inputCount, MemorySegment inputBuffers,
            int outputCount, MemorySegment outputBuffers) {
        Objects.requireNonNull(executable, "executable");
        Objects.requireNonNull(inputBuffers, "inputBuffers");
        Objects.requireNonNull(outputBuffers, "outputBuffers");
        checkExecutableStatus(EXECUTABLE_RUN_OPERATION, runExecutableNative(
                executable, inputCount, inputBuffers, outputCount, outputBuffers));
    }

    /**
     * Returns the raw status from one already validated executable run invocation.
     *
     * @param executable non-null live executable whose ownership remains with the caller
     * @param inputCount positive validated feed count
     * @param inputBuffers exact readable feed-address segment
     * @param outputCount positive validated target count
     * @param outputBuffers exact readable target-address segment
     * @return exact raw native status
     * @throws RuntimeException if the native invocation itself fails
     */
    abstract int runExecutableNative(
            Handle executable, int inputCount, MemorySegment inputBuffers,
            int outputCount, MemorySegment outputBuffers);

    /**
     * Compiles one fixed custom FLOAT32 NEG pipeline specialized to an element count.
     *
     * <p>The ABI carries {@code elementCount} as {@code uint64_t}. The custom kernel accepts only
     * {@code 1..UINT32_MAX}; zero or a larger non-negative carrier value fails with native
     * {@link Status#UNSUPPORTED_SHAPE} rather than narrowing or selecting another route.</p>
     *
     * @param context non-null live context whose ownership remains with the caller
     * @param elementCount unsigned 64-bit ABI value represented by a non-negative Java
     *     {@code long}; the successful custom domain is {@code 1..UINT32_MAX}
     * @return a fresh non-null opaque custom-pipeline handle owned by the caller
     * @throws IllegalArgumentException if {@code elementCount} is negative
     * @throws RuntimeException if the count is outside the native custom domain, construction
     *     fails, or the output-cell contract is violated
     */
    final Handle createNegKernelPipeline(Handle context, long elementCount) {
        Objects.requireNonNull(context, "context");
        if (elementCount < 0L) {
            throw new IllegalArgumentException("elementCount must be non-negative");
        }
        NativeCreateResult result = Objects.requireNonNull(
                createNegKernelPipelineNative(context, elementCount),
                "native NEG kernel pipeline create result");
        return finishCreate(
                NEG_KERNEL_PIPELINE_CREATE_OPERATION,
                result,
                this::releaseNegKernelPipelineNative);
    }

    /**
     * Returns the raw status/output-cell result of custom-pipeline creation.
     *
     * @param context non-null live context whose ownership remains with the caller
     * @param elementCount exact {@code uint64_t} carrier value represented by a non-negative Java
     *     {@code long}; native code enforces the successful {@code 1..UINT32_MAX} domain
     * @return non-null raw status and nullable output-cell handle
     * @throws RuntimeException if the native invocation itself fails
     */
    abstract NativeCreateResult createNegKernelPipelineNative(Handle context, long elementCount);

    /**
     * Consumes one live custom-NEG pipeline handle exactly once.
     *
     * @param pipeline non-null live custom-pipeline handle owned by the caller
     * @throws NullPointerException if {@code pipeline} is {@code null}
     * @throws RuntimeException if native release fails
     */
    final void releaseNegKernelPipeline(Handle pipeline) {
        Objects.requireNonNull(pipeline, "pipeline");
        checkExecutableStatus(NEG_KERNEL_PIPELINE_RELEASE_OPERATION,
                releaseNegKernelPipelineNative(pipeline));
    }

    /**
     * Returns the raw status of one custom-pipeline release invocation.
     *
     * @param pipeline non-null live custom-pipeline handle to consume
     * @return exact raw native status
     * @throws RuntimeException if the native invocation itself fails
     */
    abstract int releaseNegKernelPipelineNative(Handle pipeline);

    /**
     * Executes one custom NEG dispatch through direct input and assigned output buffer handles.
     *
     * <p>The matching native function submits one compute command and waits synchronously. It
     * writes the supplied output buffer directly and performs no explicit host staging or
     * intermediate output copy.</p>
     *
     * @param pipeline non-null live custom pipeline whose ownership remains with the caller
     * @param inputBuffer non-null live direct input buffer
     * @param outputBuffer non-null live direct output buffer
     * @throws NullPointerException if an argument is {@code null}
     * @throws RuntimeException if native validation or execution fails
     */
    final void runNegKernelPipeline(Handle pipeline, Handle inputBuffer, Handle outputBuffer) {
        Objects.requireNonNull(pipeline, "pipeline");
        Objects.requireNonNull(inputBuffer, "inputBuffer");
        Objects.requireNonNull(outputBuffer, "outputBuffer");
        checkExecutableStatus(NEG_KERNEL_PIPELINE_RUN_OPERATION,
                runNegKernelPipelineNative(pipeline, inputBuffer, outputBuffer));
    }

    /**
     * Returns the raw status of one custom NEG dispatch invocation.
     *
     * @param pipeline non-null live custom pipeline
     * @param inputBuffer non-null live direct input buffer
     * @param outputBuffer non-null live direct output buffer
     * @return exact raw native status
     * @throws RuntimeException if the native invocation itself fails
     */
    abstract int runNegKernelPipelineNative(
            Handle pipeline, Handle inputBuffer, Handle outputBuffer);

    /**
     * Ends the production symbol-lookup lifetime after every native handle has been released.
     *
     * <p>Production close is thread-safe and idempotent. Test implementations must preserve the
     * same ownership boundary but may expose deterministic injected failures.</p>
     *
     * @throws RuntimeException if lookup cleanup fails
     * @throws Error if lookup cleanup reports an error
     */
    @Override
    public abstract void close();

    /** Opaque non-null FFM carrier for one context, buffer, executable, or pipeline handle. */
    static final class Handle {
        private final MemorySegment carrier;

        /**
         * Retains one non-null, non-zero opaque carrier.
         *
         * @param carrier the native address carrier; must not be {@code null} or zero
         * @throws NullPointerException if {@code carrier} is {@code null}
         * @throws IllegalArgumentException if {@code carrier} has address zero
         */
        Handle(MemorySegment carrier) {
            this.carrier = Objects.requireNonNull(carrier, "carrier");
            if (carrier.address() == 0L) {
                throw new IllegalArgumentException("carrier must not be null");
            }
        }

        /**
         * Returns the package-private FFM carrier for a downcall.
         *
         * @return the exact non-null, non-zero carrier retained by this handle
         */
        MemorySegment carrier() {
            return carrier;
        }
    }

    /**
     * Carries the exact native create status and nullable output-cell handle.
     *
     * @param statusCode exact signed native status
     * @param handle output-cell handle, or {@code null} when native left the cell empty
     */
    record NativeCreateResult(int statusCode, Handle handle) {}

    private Handle finishExecutableCreate(NativeCreateResult result) {
        return finishCreate(EXECUTABLE_CREATE_OPERATION, result, this::releaseExecutableNative);
    }

    private Handle finishCreate(
            String operation, NativeCreateResult result, NativeHandleRelease release) {
        int status = result.statusCode();
        Handle handle = result.handle();
        if (status == 0 && handle != null) return handle;
        RuntimeException failure = status == 0
                ? new IllegalStateException(
                        operation + " returned OK with a null handle")
                : new NativeFailure(operation, status);
        if (handle != null) {
            try {
                checkExecutableStatus(
                        operation + " malformed-handle cleanup", release.release(handle));
            } catch (RuntimeException | Error cleanup) {
                if (cleanup != failure) failure.addSuppressed(cleanup);
            }
            if (status != 0) {
                IllegalStateException contractFailure = new IllegalStateException(
                        operation + " returned failure with a non-null handle",
                        failure);
                for (Throwable suppressed : failure.getSuppressed()) {
                    contractFailure.addSuppressed(suppressed);
                }
                throw contractFailure;
            }
        }
        throw failure;
    }

    @FunctionalInterface
    private interface NativeHandleRelease {
        int release(Handle handle);
    }

    private static void checkExecutableStatus(String operation, int status) {
        if (status != 0) throw new NativeFailure(operation, status);
    }

    /** Stable version-four non-success status meanings. */
    enum Status {
        INVALID_ARGUMENT(1),
        NO_DEVICE(2),
        NO_COMMAND_QUEUE(3),
        ALLOCATION_FAILED(4),
        RANGE_OUT_OF_BOUNDS(5),
        COPY_FAILED(6),
        INTERNAL_ERROR(7),
        UNSUPPORTED_SHAPE(8),
        GRAPH_COMPILATION_FAILED(9),
        INCOMPATIBLE_RESOURCE(10),
        EXECUTION_FAILED(11),
        KERNEL_COMPILATION_FAILED(12);

        private final int code;

        Status(int code) {
            this.code = code;
        }

        static Status fromCode(int code) {
            for (Status status : values()) {
                if (status.code == code) {
                    return status;
                }
            }
            return null;
        }
    }

    /** Unchecked failure retaining stable native operation and status facts. */
    static final class NativeFailure extends RuntimeException {
        private final String operation;
        private final int statusCode;
        private final Status status;

        /**
         * Creates a known- or unknown-status native failure.
         *
         * @param operation stable non-null ABI operation name
         * @param statusCode exact signed status value returned by native code
         * @throws NullPointerException if {@code operation} is {@code null}
         */
        NativeFailure(String operation, int statusCode) {
            super(message(operation, statusCode));
            this.operation = Objects.requireNonNull(operation, "operation");
            this.statusCode = statusCode;
            this.status = Status.fromCode(statusCode);
        }

        /** @return the stable non-null ABI operation name */
        String operation() {
            return operation;
        }

        /** @return the exact signed status integer returned by native code */
        int statusCode() {
            return statusCode;
        }

        /** @return the known status, or {@code null} when native code returned an unknown value */
        Status status() {
            return status;
        }

        private static String message(String operation, int statusCode) {
            Status status = Status.fromCode(statusCode);
            return operation + " failed with "
                    + (status == null ? "unknown native status " + statusCode
                            : "native status " + status + " (" + statusCode + ")");
        }
    }

    /** Exact Java preflight for the version-eight typed MPSGraph executable-create schema. */
    static final class MpsGraphExecutableAbi {
        private static final int MAX_RANK = 16;

        private MpsGraphExecutableAbi() {}

        static void validateCreate(
                NumericalProfile numericalProfile,
                int[] valueRanks,
                long[] valueDimensions,
                MetalMpsGraphProgram graphProgram,
                int[] feeds,
                int[] targets) {
            Objects.requireNonNull(numericalProfile, "numericalProfile");
            Objects.requireNonNull(valueRanks, "valueRanks");
            Objects.requireNonNull(valueDimensions, "valueDimensions");
            Objects.requireNonNull(graphProgram, "graphProgram");
            Objects.requireNonNull(feeds, "feedValueIndices");
            Objects.requireNonNull(targets, "targetValueIndices");
            int valueCount = valueRanks.length;
            if (valueCount == 0) {
                throw new IllegalArgumentException("Metal MPSGraph value count must be positive");
            }
            int dimensionCount;
            try {
                dimensionCount = Math.multiplyExact(valueCount, MAX_RANK);
            } catch (ArithmeticException overflow) {
                throw new IllegalArgumentException(
                        "Metal MPSGraph dimension-table cardinality overflows", overflow);
            }
            if (valueDimensions.length != dimensionCount) {
                throw new IllegalArgumentException(
                        "Metal MPSGraph dimensions must contain exactly valueCount * 16 cells");
            }
            if (graphProgram.nodes().isEmpty()) {
                throw new IllegalArgumentException("Metal MPSGraph node count must be positive");
            }
            if (feeds.length == 0 || targets.length == 0) {
                throw new IllegalArgumentException(
                        "Metal MPSGraph feed and target counts must be positive");
            }

            for (int value = 0; value < valueCount; value++) {
                int rank = valueRanks[value];
                if (rank < 0 || rank > MAX_RANK) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph value rank must be in 0..16 at index " + value);
                }
                long elements = 1L;
                int row = value * MAX_RANK;
                try {
                    for (int axis = 0; axis < rank; axis++) {
                        long dimension = valueDimensions[row + axis];
                        if (dimension <= 0L) {
                            throw new IllegalArgumentException(
                                    "Metal MPSGraph used dimensions must be positive");
                        }
                        elements = Math.multiplyExact(elements, dimension);
                    }
                    Math.multiplyExact(elements, Float.BYTES);
                } catch (ArithmeticException overflow) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph FLOAT32 geometry overflows at value " + value,
                            overflow);
                }
                for (int axis = rank; axis < MAX_RANK; axis++) {
                    if (valueDimensions[row + axis] != 0L) {
                        throw new IllegalArgumentException(
                                "Metal MPSGraph unused dimension cells must be zero");
                    }
                }
            }

            MetalMpsGraphProgram.ValueState[] states =
                    new MetalMpsGraphProgram.ValueState[valueCount];
            java.util.Arrays.fill(states, MetalMpsGraphProgram.ValueState.UNAVAILABLE);
            boolean[] used = new boolean[valueCount];
            boolean[] produced = new boolean[valueCount];
            boolean[] localTranspose = new boolean[valueCount];
            for (int feed : feeds) {
                requireIndex(feed, valueCount, "feed");
                if (states[feed] != MetalMpsGraphProgram.ValueState.UNAVAILABLE) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph feed indices must be unique");
                }
                if (valueRanks[feed] == 0) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph rank-zero values cannot be feeds");
                }
                states[feed] = MetalMpsGraphProgram.ValueState.CANONICAL;
                used[feed] = true;
            }
            for (MetalMpsGraphProgram.Node node : graphProgram.nodes()) {
                if (!profileAllows(numericalProfile, node.kind())) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph node kind is incompatible with numerical profile");
                }
            }
            for (MetalMpsGraphProgram.Node node : graphProgram.nodes()) {
                int left = node.firstInputIndex();
                int right = node.secondInputIndex();
                int output = node.outputIndex();
                requireIndex(left, valueCount, "first node input");
                requireIndex(output, valueCount, "node output");
                if (valueRanks[left] == 0) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph rank-zero results cannot be node inputs");
                }
                if (!node.kind().accepts(states[left])) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph node input value state is unavailable or incompatible");
                }
                if (states[output] != MetalMpsGraphProgram.ValueState.UNAVAILABLE) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph node outputs must be unique and not feeds");
                }
                switch (node.kind()) {
                    case NEG, ABS, CONTIGUOUS -> requireShape(
                            sameShape(left, output, valueRanks, valueDimensions),
                            node.kind() + " input/output shapes must match exactly");
                    case ADD, SUB, MUL, DIV -> {
                        requireIndex(right, valueCount, "second node input");
                        if (valueRanks[right] == 0
                                || !node.kind().accepts(states[right])) {
                            throw new IllegalArgumentException(
                                    "Metal MPSGraph second node input must be positive-rank canonical");
                        }
                        requireShape(
                                broadcastsTo(left, right, output, valueRanks, valueDimensions),
                                "binary output must equal exact right-aligned broadcast");
                        used[right] = true;
                    }
                    case RESHAPE -> {
                        requireShape(
                                sameElementCount(left, output, valueRanks, valueDimensions),
                                "RESHAPE input/output element counts must match");
                        requireShape(
                                targetMatches(node, output, valueRanks, valueDimensions),
                                "RESHAPE target attributes must equal the output shape");
                    }
                    case EXPAND -> {
                        requireShape(
                                expandsTo(left, output, valueRanks, valueDimensions),
                                "EXPAND output must be an exact right-aligned expansion");
                        requireShape(
                                targetMatches(node, output, valueRanks, valueDimensions),
                                "EXPAND target attributes must equal the output shape");
                    }
                    case PERMUTE -> {
                        boolean matches = permutationMatches(
                                node, left, output, valueRanks, valueDimensions);
                        requireShape(matches, "PERMUTE attributes and output shape disagree");
                        long[] attributes = node.attributeValues();
                        boolean exactLocalTranspose =
                                states[left] == MetalMpsGraphProgram.ValueState.CANONICAL
                                        && valueRanks[left] == 2
                                        && valueRanks[output] == 2
                                        && node.attributeCount() == 2
                                        && attributes[0] == 1L
                                        && attributes[1] == 0L;
                        localTranspose[output] = exactLocalTranspose;
                    }
                    case EXPAND_DIMS -> requireShape(
                            expandDimsMatches(node, left, output, valueRanks, valueDimensions),
                            "EXPAND_DIMS axis and output shape disagree");
                    case SQUEEZE -> requireShape(
                            squeezeMatches(node, left, output, valueRanks, valueDimensions),
                            "SQUEEZE axis and output shape disagree");
                    case SUM, MEAN -> requireShape(
                            reductionMatches(node, left, output, valueRanks, valueDimensions),
                            "reduction attributes, count, and output shape disagree");
                    case MATMUL -> {
                        requireIndex(right, valueCount, "second node input");
                        if (valueRanks[right] == 0
                                || !node.kind().accepts(states[right])) {
                            throw new IllegalArgumentException(
                                    "Metal MATMUL second input must be positive-rank canonical"
                                            + " or an affine view");
                        }
                        if ((states[left] == MetalMpsGraphProgram.ValueState.AFFINE_VIEW
                                        && !localTranspose[left])
                                || (states[right]
                                                == MetalMpsGraphProgram.ValueState.AFFINE_VIEW
                                        && !localTranspose[right])) {
                            throw new IllegalArgumentException(
                                    "Metal MATMUL affine inputs must be authenticated local"
                                            + " rank-two transposes");
                        }
                        requireShape(
                                matmulMatches(
                                        left, right, output, valueRanks, valueDimensions),
                                "MATMUL shapes must be exact positive rank-two contraction");
                        used[right] = true;
                    }
                }
                if (valueRanks[output] == 0
                        && node.kind() != MetalMpsGraphProgram.NodeKind.SUM
                        && node.kind() != MetalMpsGraphProgram.NodeKind.MEAN) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph rank-zero values must be locally produced reductions");
                }
                states[output] = node.kind().outputState();
                used[left] = true;
                used[output] = true;
                produced[output] = true;
            }
            boolean[] targeted = new boolean[valueCount];
            for (int target : targets) {
                requireIndex(target, valueCount, "target");
                if (!produced[target]
                        || states[target] == MetalMpsGraphProgram.ValueState.UNAVAILABLE) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph target must be a node-produced value");
                }
                if (targeted[target]) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph target indices must be unique");
                }
                targeted[target] = true;
            }
            for (int value = 0; value < valueCount; value++) {
                if (valueRanks[value] == 0 && (!produced[value] || !targeted[value])) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph rank-zero reduction must be a direct target");
                }
            }
            for (int value = 0; value < valueCount; value++) {
                if (!used[value]) {
                    throw new IllegalArgumentException(
                            "Metal MPSGraph value table contains an unused index: " + value);
                }
            }
        }

        static void validateRun(
                int expectedInputs,
                int inputCount,
                MemorySegment inputBuffers,
                int expectedOutputs,
                int outputCount,
                MemorySegment outputBuffers) {
            Objects.requireNonNull(inputBuffers, "inputBuffers");
            Objects.requireNonNull(outputBuffers, "outputBuffers");
            if (inputCount != expectedInputs || outputCount != expectedOutputs) {
                throw new IllegalArgumentException(
                        "Metal MPSGraph run counts must match the compiled feeds and targets");
            }
            validateAddressSegment(inputBuffers, inputCount, "inputBuffers");
            validateAddressSegment(outputBuffers, outputCount, "outputBuffers");
            for (int input = 0; input < inputCount; input++) {
                long inputAddress = inputBuffers.getAtIndex(ADDRESS, input).address();
                for (int output = 0; output < outputCount; output++) {
                    if (inputAddress == outputBuffers.getAtIndex(ADDRESS, output).address()) {
                        throw new IllegalArgumentException(
                                "Metal MPSGraph input and output native buffers must not alias");
                    }
                }
            }
        }

        private static void validateAddressSegment(
                MemorySegment segment, int count, String name) {
            if (!segment.scope().isAlive() || !segment.isAccessibleBy(Thread.currentThread())) {
                throw new IllegalStateException(name + " is not alive and accessible");
            }
            if (!segment.isNative()) {
                throw new IllegalArgumentException(name + " must be native memory");
            }
            long expectedBytes = Math.multiplyExact((long) count, ADDRESS.byteSize());
            if (segment.byteSize() != expectedBytes) {
                throw new IllegalArgumentException(
                        name + " byte size must exactly match its handle count");
            }
            for (int index = 0; index < count; index++) {
                if (segment.getAtIndex(ADDRESS, index).address() == 0L) {
                    throw new IllegalArgumentException(name + " contains a null handle");
                }
            }
        }

        private static void requireIndex(int index, int valueCount, String role) {
            if (index < 0 || index >= valueCount) {
                throw new IllegalArgumentException(
                        "Metal MPSGraph " + role + " index is out of range");
            }
        }

        private static boolean profileAllows(
                NumericalProfile numericalProfile, MetalMpsGraphProgram.NodeKind kind) {
            return switch (numericalProfile) {
                case STRICT_IEEE -> switch (kind) {
                    case NEG, ABS, RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE,
                            CONTIGUOUS -> true;
                    case ADD, SUB, MUL, DIV, SUM, MEAN, MATMUL -> false;
                };
                case ACCELERATOR -> true;
            };
        }

        private static boolean matmulMatches(
                int left,
                int right,
                int output,
                int[] ranks,
                long[] dimensions) {
            if (ranks[left] != 2 || ranks[right] != 2 || ranks[output] != 2) {
                return false;
            }
            int leftRow = left * MAX_RANK;
            int rightRow = right * MAX_RANK;
            int outputRow = output * MAX_RANK;
            return dimensions[leftRow + 1] == dimensions[rightRow]
                    && dimensions[outputRow] == dimensions[leftRow]
                    && dimensions[outputRow + 1] == dimensions[rightRow + 1];
        }

        private static boolean reductionMatches(
                MetalMpsGraphProgram.Node node,
                int input,
                int output,
                int[] ranks,
                long[] dimensions) {
            int inputRank = ranks[input];
            int outputRank = ranks[output];
            long[] values = node.attributeValues();
            MetalMpsGraphProgram.ReductionForm form;
            try {
                form = MetalMpsGraphProgram.ReductionForm.fromWireIdentity(node.axis());
            } catch (IllegalArgumentException malformed) {
                return false;
            }
            if (form == MetalMpsGraphProgram.ReductionForm.SUM_TO_SHAPE) {
                if (node.kind() != MetalMpsGraphProgram.NodeKind.SUM
                        || node.reserved() != 0
                        || node.attributeCount() != outputRank
                        || values.length != outputRank
                        || outputRank > inputRank) {
                    return false;
                }
                int inputRow = input * MAX_RANK;
                int outputRow = output * MAX_RANK;
                int padding = inputRank - outputRank;
                long count = 1L;
                try {
                    for (int axis = 0; axis < inputRank; axis++) {
                        long source = dimensions[inputRow + axis];
                        if (axis < padding) {
                            count = Math.multiplyExact(count, source);
                        } else {
                            long target = values[axis - padding];
                            if (target != dimensions[outputRow + axis - padding]
                                    || (target != 1L && target != source)) {
                                return false;
                            }
                            if (target == 1L && source != 1L) {
                                count = Math.multiplyExact(count, source);
                            }
                        }
                    }
                } catch (ArithmeticException overflow) {
                    return false;
                }
                return count > 0L;
            }

            if (form == MetalMpsGraphProgram.ReductionForm.FULL) {
                if (node.attributeCount() != 0 || values.length != 0
                        || node.reserved() != 0 || outputRank != 0) {
                    return false;
                }
                try {
                    return elementCount(input, ranks, dimensions) > 0L;
                } catch (ArithmeticException overflow) {
                    return false;
                }
            }

            if (form == MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS
                    && node.attributeCount() != 1) {
                return false;
            }
            if (form != MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS
                    && form != MetalMpsGraphProgram.ReductionForm.MULTI_AXIS) {
                return false;
            }
            boolean keep = node.reserved() == 1;
            if (node.reserved() < 0 || node.reserved() > 1
                    || node.attributeCount() != values.length) {
                return false;
            }
            boolean[] reduced = new boolean[inputRank];
            long count = 1L;
            int inputRow = input * MAX_RANK;
            try {
                for (long value : values) {
                    if (value < 0L || value >= inputRank || reduced[(int) value]) {
                        return false;
                    }
                    reduced[(int) value] = true;
                    count = Math.multiplyExact(count, dimensions[inputRow + (int) value]);
                }
            } catch (ArithmeticException overflow) {
                return false;
            }
            int expectedRank = keep ? inputRank : inputRank - values.length;
            if (outputRank != expectedRank || count <= 0L) return false;
            int outputRow = output * MAX_RANK;
            for (int source = 0, target = 0; source < inputRank; source++) {
                if (reduced[source]) {
                    if (keep && dimensions[outputRow + target++] != 1L) return false;
                } else if (dimensions[outputRow + target++]
                        != dimensions[inputRow + source]) {
                    return false;
                }
            }
            return true;
        }

        private static boolean sameShape(
                int first, int second, int[] ranks, long[] dimensions) {
            int rank = ranks[first];
            if (rank != ranks[second]) return false;
            int firstRow = first * MAX_RANK;
            int secondRow = second * MAX_RANK;
            for (int axis = 0; axis < rank; axis++) {
                if (dimensions[firstRow + axis] != dimensions[secondRow + axis]) return false;
            }
            return true;
        }

        private static void requireShape(boolean condition, String message) {
            if (!condition) {
                throw new IllegalArgumentException("Metal MPSGraph " + message);
            }
        }

        private static boolean sameElementCount(
                int first, int second, int[] ranks, long[] dimensions) {
            return elementCount(first, ranks, dimensions)
                    == elementCount(second, ranks, dimensions);
        }

        private static long elementCount(int value, int[] ranks, long[] dimensions) {
            long elements = 1L;
            int row = value * MAX_RANK;
            for (int axis = 0; axis < ranks[value]; axis++) {
                elements = Math.multiplyExact(elements, dimensions[row + axis]);
            }
            return elements;
        }

        private static boolean targetMatches(
                MetalMpsGraphProgram.Node node,
                int output,
                int[] ranks,
                long[] dimensions) {
            if (node.attributeCount() != ranks[output]) {
                return false;
            }
            long[] target = node.attributeValues();
            int row = output * MAX_RANK;
            for (int axis = 0; axis < target.length; axis++) {
                if (target[axis] != dimensions[row + axis]) {
                    return false;
                }
            }
            return true;
        }

        private static boolean expandsTo(
                int input, int output, int[] ranks, long[] dimensions) {
            int inputRank = ranks[input];
            int outputRank = ranks[output];
            if (inputRank > outputRank) {
                return false;
            }
            int inputRow = input * MAX_RANK;
            int outputRow = output * MAX_RANK;
            int padding = outputRank - inputRank;
            for (int axis = 0; axis < inputRank; axis++) {
                long source = dimensions[inputRow + axis];
                long target = dimensions[outputRow + axis + padding];
                if (source != target && source != 1L) {
                    return false;
                }
            }
            return true;
        }

        private static boolean permutationMatches(
                MetalMpsGraphProgram.Node node,
                int input,
                int output,
                int[] ranks,
                long[] dimensions) {
            int rank = ranks[input];
            if (ranks[output] != rank || node.attributeCount() != rank) {
                return false;
            }
            long[] permutation = node.attributeValues();
            int inputRow = input * MAX_RANK;
            int outputRow = output * MAX_RANK;
            for (int axis = 0; axis < rank; axis++) {
                int source = Math.toIntExact(permutation[axis]);
                if (dimensions[outputRow + axis] != dimensions[inputRow + source]) {
                    return false;
                }
            }
            return true;
        }

        private static boolean expandDimsMatches(
                MetalMpsGraphProgram.Node node,
                int input,
                int output,
                int[] ranks,
                long[] dimensions) {
            int inputRank = ranks[input];
            int axis = node.axis();
            if (ranks[output] != inputRank + 1 || axis > inputRank) {
                return false;
            }
            int inputRow = input * MAX_RANK;
            int outputRow = output * MAX_RANK;
            for (int outputAxis = 0; outputAxis < ranks[output]; outputAxis++) {
                long expected = outputAxis == axis
                        ? 1L
                        : dimensions[inputRow + (outputAxis < axis
                                ? outputAxis : outputAxis - 1)];
                if (dimensions[outputRow + outputAxis] != expected) {
                    return false;
                }
            }
            return true;
        }

        private static boolean squeezeMatches(
                MetalMpsGraphProgram.Node node,
                int input,
                int output,
                int[] ranks,
                long[] dimensions) {
            int inputRank = ranks[input];
            int axis = node.axis();
            if (ranks[output] != inputRank - 1
                    || axis >= inputRank
                    || dimensions[input * MAX_RANK + axis] != 1L) {
                return false;
            }
            int inputRow = input * MAX_RANK;
            int outputRow = output * MAX_RANK;
            for (int source = 0, target = 0; source < inputRank; source++) {
                if (source != axis
                        && dimensions[inputRow + source] != dimensions[outputRow + target++]) {
                    return false;
                }
            }
            return true;
        }

        private static boolean broadcastsTo(
                int left,
                int right,
                int output,
                int[] ranks,
                long[] dimensions) {
            int leftRank = ranks[left];
            int rightRank = ranks[right];
            int outputRank = ranks[output];
            if (outputRank != Math.max(leftRank, rightRank)) {
                return false;
            }
            int leftRow = left * MAX_RANK;
            int rightRow = right * MAX_RANK;
            int outputRow = output * MAX_RANK;
            int leftPadding = outputRank - leftRank;
            int rightPadding = outputRank - rightRank;
            for (int axis = 0; axis < outputRank; axis++) {
                long leftDimension = axis < leftPadding
                        ? 1L : dimensions[leftRow + axis - leftPadding];
                long rightDimension = axis < rightPadding
                        ? 1L : dimensions[rightRow + axis - rightPadding];
                if (leftDimension != rightDimension
                        && leftDimension != 1L && rightDimension != 1L) {
                    return false;
                }
                if (dimensions[outputRow + axis]
                        != Math.max(leftDimension, rightDimension)) {
                    return false;
                }
            }
            return true;
        }

    }

    /** Production JDK Foreign Function and Memory binding of the exact thirteen-symbol ABI. */
    private static final class Ffm extends MetalNativeApi {
        private static final String VERSION = "synaptik_metal_foundation_abi_version";
        private static final String CONTEXT_CREATE = "synaptik_metal_context_create";
        private static final String CONTEXT_RELEASE = "synaptik_metal_context_release";
        private static final String BUFFER_CREATE = "synaptik_metal_buffer_create";
        private static final String BUFFER_RELEASE = "synaptik_metal_buffer_release";
        private static final String BUFFER_UPLOAD = "synaptik_metal_buffer_upload";
        private static final String BUFFER_DOWNLOAD = "synaptik_metal_buffer_download";
        private static final String EXECUTABLE_CREATE =
                EXECUTABLE_CREATE_OPERATION;
        private static final String EXECUTABLE_RELEASE =
                EXECUTABLE_RELEASE_OPERATION;
        private static final String EXECUTABLE_RUN =
                EXECUTABLE_RUN_OPERATION;
        private static final String NEG_KERNEL_PIPELINE_CREATE =
                NEG_KERNEL_PIPELINE_CREATE_OPERATION;
        private static final String NEG_KERNEL_PIPELINE_RELEASE =
                NEG_KERNEL_PIPELINE_RELEASE_OPERATION;
        private static final String NEG_KERNEL_PIPELINE_RUN =
                NEG_KERNEL_PIPELINE_RUN_OPERATION;

        private final Arena lookupArena;
        private final MethodHandle contextCreate;
        private final MethodHandle contextRelease;
        private final MethodHandle bufferCreate;
        private final MethodHandle bufferRelease;
        private final MethodHandle bufferUpload;
        private final MethodHandle bufferDownload;
        private final MethodHandle executableCreate;
        private final MethodHandle executableRelease;
        private final MethodHandle executableRun;
        private final MethodHandle negKernelPipelineCreate;
        private final MethodHandle negKernelPipelineRelease;
        private final MethodHandle negKernelPipelineRun;
        private final AtomicBoolean closed = new AtomicBoolean();

        private Ffm(
                Arena lookupArena,
                MethodHandle contextCreate,
                MethodHandle contextRelease,
                MethodHandle bufferCreate,
                MethodHandle bufferRelease,
                MethodHandle bufferUpload,
                MethodHandle bufferDownload,
                MethodHandle executableCreate,
                MethodHandle executableRelease,
                MethodHandle executableRun,
                MethodHandle negKernelPipelineCreate,
                MethodHandle negKernelPipelineRelease,
                MethodHandle negKernelPipelineRun) {
            this.lookupArena = lookupArena;
            this.contextCreate = contextCreate;
            this.contextRelease = contextRelease;
            this.bufferCreate = bufferCreate;
            this.bufferRelease = bufferRelease;
            this.bufferUpload = bufferUpload;
            this.bufferDownload = bufferDownload;
            this.executableCreate = executableCreate;
            this.executableRelease = executableRelease;
            this.executableRun = executableRun;
            this.negKernelPipelineCreate = negKernelPipelineCreate;
            this.negKernelPipelineRelease = negKernelPipelineRelease;
            this.negKernelPipelineRun = negKernelPipelineRun;
        }

        static Ffm open(Path path) {
            Arena arena = Arena.ofShared();
            try {
                SymbolLookup lookup = SymbolLookup.libraryLookup(path, arena);
                Linker linker = Linker.nativeLinker();

                MethodHandle version = linker.downcallHandle(
                        require(lookup, VERSION), FunctionDescriptor.of(JAVA_INT));
                int actualVersion = invokeInt(version, VERSION);
                if (Integer.toUnsignedLong(actualVersion) != ABI_VERSION) {
                    throw new IllegalStateException(
                            "Metal foundation ABI version must be " + ABI_VERSION
                                    + " but was " + Integer.toUnsignedLong(actualVersion));
                }

                MethodHandle contextCreate = linker.downcallHandle(
                        require(lookup, CONTEXT_CREATE), FunctionDescriptor.of(JAVA_INT, ADDRESS));
                MethodHandle contextRelease = linker.downcallHandle(
                        require(lookup, CONTEXT_RELEASE), FunctionDescriptor.of(JAVA_INT, ADDRESS));
                MethodHandle bufferCreate = linker.downcallHandle(
                        require(lookup, BUFFER_CREATE),
                        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
                MethodHandle bufferRelease = linker.downcallHandle(
                        require(lookup, BUFFER_RELEASE), FunctionDescriptor.of(JAVA_INT, ADDRESS));
                MethodHandle bufferUpload = linker.downcallHandle(
                        require(lookup, BUFFER_UPLOAD),
                        FunctionDescriptor.of(
                                JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG));
                MethodHandle bufferDownload = linker.downcallHandle(
                        require(lookup, BUFFER_DOWNLOAD),
                        FunctionDescriptor.of(
                                JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS, JAVA_LONG));
                MethodHandle executableCreate = linker.downcallHandle(
                        require(lookup, EXECUTABLE_CREATE),
                        FunctionDescriptor.of(JAVA_INT,
                                ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS,
                                JAVA_INT, ADDRESS, JAVA_INT, ADDRESS,
                                JAVA_INT, ADDRESS, ADDRESS));
                MethodHandle executableRelease = linker.downcallHandle(
                        require(lookup, EXECUTABLE_RELEASE),
                        FunctionDescriptor.of(JAVA_INT, ADDRESS));
                MethodHandle executableRun = linker.downcallHandle(
                        require(lookup, EXECUTABLE_RUN),
                        FunctionDescriptor.of(
                                JAVA_INT, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS));
                MethodHandle negKernelPipelineCreate = linker.downcallHandle(
                        require(lookup, NEG_KERNEL_PIPELINE_CREATE),
                        FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_LONG, ADDRESS));
                MethodHandle negKernelPipelineRelease = linker.downcallHandle(
                        require(lookup, NEG_KERNEL_PIPELINE_RELEASE),
                        FunctionDescriptor.of(JAVA_INT, ADDRESS));
                MethodHandle negKernelPipelineRun = linker.downcallHandle(
                        require(lookup, NEG_KERNEL_PIPELINE_RUN),
                        FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS));
                return new Ffm(arena, contextCreate, contextRelease, bufferCreate, bufferRelease,
                        bufferUpload, bufferDownload, executableCreate, executableRelease,
                        executableRun, negKernelPipelineCreate, negKernelPipelineRelease,
                        negKernelPipelineRun);
            } catch (RuntimeException | Error failure) {
                closeAfterFailure(arena, failure);
                throw failure;
            }
        }

        @Override
        Handle createContext() {
            return create(CONTEXT_CREATE, contextCreate, null, 0L, contextRelease);
        }

        @Override
        void releaseContext(Handle context) {
            requireOpen();
            checkStatus(CONTEXT_RELEASE,
                    invokeIntAddress(contextRelease, CONTEXT_RELEASE, context.carrier()));
        }

        @Override
        Handle createBuffer(Handle context, long logicalByteSize) {
            requireOpen();
            if (logicalByteSize < 0L) {
                throw new IllegalArgumentException("logicalByteSize must be non-negative");
            }
            return create(BUFFER_CREATE, bufferCreate, context, logicalByteSize, bufferRelease);
        }

        @Override
        void releaseBuffer(Handle buffer) {
            requireOpen();
            checkStatus(BUFFER_RELEASE,
                    invokeIntAddress(bufferRelease, BUFFER_RELEASE, buffer.carrier()));
        }

        @Override
        void upload(Handle buffer, long bufferOffset, MemorySegment source, long byteCount) {
            requireOpen();
            int status = invokeIntCopy(bufferUpload, BUFFER_UPLOAD, buffer.carrier(), bufferOffset,
                    source, byteCount);
            checkStatus(BUFFER_UPLOAD, status);
        }

        @Override
        void download(
                Handle buffer, long bufferOffset, MemorySegment destination, long byteCount) {
            requireOpen();
            int status = invokeIntCopy(bufferDownload, BUFFER_DOWNLOAD, buffer.carrier(),
                    bufferOffset, destination, byteCount);
            checkStatus(BUFFER_DOWNLOAD, status);
        }

        @Override
        NativeCreateResult createMpsGraphExecutableNative(
                Handle context,
                int[] valueRanks,
                long[] valueDimensions,
                MetalMpsGraphProgram graphProgram,
                int[] feedValueIndices,
                int[] targetValueIndices) {
            requireOpen();
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment ranks = copyInts(arena, valueRanks);
                MemorySegment dimensions = copyLongs(arena, valueDimensions);
                MemorySegment nodes = graphProgram.encodeNative(arena);
                MemorySegment feeds = copyInts(arena, feedValueIndices);
                MemorySegment targets = copyInts(arena, targetValueIndices);
                MemorySegment output = arena.allocate(ADDRESS);
                output.set(ADDRESS, 0L, MemorySegment.NULL);
                int status = invokeCreateExecutable(
                        executableCreate,
                        context.carrier(),
                        MetalMpsGraphProgram.SCHEMA_VERSION,
                        valueRanks.length,
                        ranks,
                        dimensions,
                        graphProgram.nodes().size(),
                        nodes,
                        feedValueIndices.length,
                        feeds,
                        targetValueIndices.length,
                        targets,
                        output);
                MemorySegment carrier = output.get(ADDRESS, 0L);
                return new NativeCreateResult(status,
                        carrier.address() == 0L ? null : new Handle(carrier));
            }
        }

        @Override
        int releaseExecutableNative(Handle executable) {
            requireOpen();
            return invokeIntAddress(
                    executableRelease, EXECUTABLE_RELEASE, executable.carrier());
        }

        @Override
        int runExecutableNative(Handle executable, int inputCount, MemorySegment inputBuffers,
                int outputCount, MemorySegment outputBuffers) {
            requireOpen();
            return invokeRunExecutable(executableRun, executable.carrier(),
                    inputCount, inputBuffers, outputCount, outputBuffers);
        }

        @Override
        NativeCreateResult createNegKernelPipelineNative(Handle context, long elementCount) {
            requireOpen();
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment output = arena.allocate(ADDRESS);
                output.set(ADDRESS, 0L, MemorySegment.NULL);
                int status = invokeIntCreateBuffer(
                        negKernelPipelineCreate, NEG_KERNEL_PIPELINE_CREATE,
                        context.carrier(), elementCount, output);
                MemorySegment carrier = output.get(ADDRESS, 0L);
                return new NativeCreateResult(
                        status, carrier.address() == 0L ? null : new Handle(carrier));
            }
        }

        @Override
        int releaseNegKernelPipelineNative(Handle pipeline) {
            requireOpen();
            return invokeIntAddress(negKernelPipelineRelease,
                    NEG_KERNEL_PIPELINE_RELEASE, pipeline.carrier());
        }

        @Override
        int runNegKernelPipelineNative(
                Handle pipeline, Handle inputBuffer, Handle outputBuffer) {
            requireOpen();
            return invokeIntThreeAddresses(negKernelPipelineRun, NEG_KERNEL_PIPELINE_RUN,
                    pipeline.carrier(), inputBuffer.carrier(), outputBuffer.carrier());
        }

        @Override
        public void close() {
            if (closed.compareAndSet(false, true)) {
                lookupArena.close();
            }
        }

        private Handle create(
                String operation,
                MethodHandle create,
                Handle context,
                long logicalByteSize,
                MethodHandle malformedHandleRelease) {
            requireOpen();
            try (Arena outputArena = Arena.ofConfined()) {
                MemorySegment output = outputArena.allocate(ADDRESS);
                output.set(ADDRESS, 0L, MemorySegment.NULL);
                int status = context == null
                        ? invokeIntAddress(create, operation, output)
                        : invokeIntCreateBuffer(
                                create, operation, context.carrier(), logicalByteSize, output);
                return finishCreate(operation, status, output.get(ADDRESS, 0L),
                        malformedHandleRelease);
            }
        }

        private static Handle finishCreate(String operation, int status, MemorySegment carrier,
                MethodHandle malformedHandleRelease) {
            if (status == 0 && carrier.address() != 0L) return new Handle(carrier);
            RuntimeException failure = status == 0
                    ? new IllegalStateException(operation + " returned OK with a null handle")
                    : new NativeFailure(operation, status);
            if (carrier.address() != 0L) {
                try {
                    checkStatus(operation + " malformed-handle cleanup",
                            invokeIntAddress(malformedHandleRelease,
                                    operation + " malformed-handle cleanup", carrier));
                } catch (RuntimeException | Error cleanup) {
                    if (cleanup != failure) failure.addSuppressed(cleanup);
                }
                if (status != 0) {
                    IllegalStateException contractFailure = new IllegalStateException(
                            operation + " returned failure with a non-null handle", failure);
                    for (Throwable suppressed : failure.getSuppressed()) {
                        contractFailure.addSuppressed(suppressed);
                    }
                    throw contractFailure;
                }
            }
            throw failure;
        }

        private static MemorySegment copyInts(Arena arena, int[] values) {
            MemorySegment result = arena.allocate(JAVA_INT, values.length);
            for (int index = 0; index < values.length; index++) {
                result.setAtIndex(JAVA_INT, index, values[index]);
            }
            return result;
        }

        private static MemorySegment copyLongs(Arena arena, long[] values) {
            MemorySegment result = arena.allocate(JAVA_LONG, values.length);
            for (int index = 0; index < values.length; index++) {
                result.setAtIndex(JAVA_LONG, index, values[index]);
            }
            return result;
        }

        private void requireOpen() {
            if (closed.get()) {
                throw new IllegalStateException("Metal native API is closed");
            }
        }

        private static MemorySegment require(SymbolLookup lookup, String symbol) {
            return lookup.find(symbol)
                    .orElseThrow(() -> new IllegalStateException(
                            "Missing required Metal foundation symbol: " + symbol));
        }

        private static void checkStatus(String operation, int status) {
            if (status != 0) {
                throw new NativeFailure(operation, status);
            }
        }

        private static int invokeInt(MethodHandle handle, String operation) {
            try {
                return (int) handle.invokeExact();
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(operation + " invocation failed", failure);
            }
        }

        private static int invokeIntAddress(
                MethodHandle handle, String operation, MemorySegment value) {
            try {
                return (int) handle.invokeExact(value);
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(operation + " invocation failed", failure);
            }
        }

        private static int invokeIntCreateBuffer(
                MethodHandle handle,
                String operation,
                MemorySegment context,
                long logicalByteSize,
                MemorySegment output) {
            try {
                return (int) handle.invokeExact(context, logicalByteSize, output);
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(operation + " invocation failed", failure);
            }
        }

        private static int invokeIntCopy(
                MethodHandle handle,
                String operation,
                MemorySegment buffer,
                long bufferOffset,
                MemorySegment bytes,
                long byteCount) {
            try {
                return (int) handle.invokeExact(buffer, bufferOffset, bytes, byteCount);
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(operation + " invocation failed", failure);
            }
        }

        private static int invokeIntThreeAddresses(
                MethodHandle handle,
                String operation,
                MemorySegment first,
                MemorySegment second,
                MemorySegment third) {
            try {
                return (int) handle.invokeExact(first, second, third);
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(operation + " invocation failed", failure);
            }
        }

        private static int invokeCreateExecutable(
                MethodHandle handle,
                MemorySegment context,
                int nodeSchemaVersion,
                int valueCount,
                MemorySegment ranks,
                MemorySegment dimensions,
                int nodeCount,
                MemorySegment nodes,
                int feedCount,
                MemorySegment feeds,
                int targetCount,
                MemorySegment targets,
                MemorySegment output) {
            try {
                return (int) handle.invokeExact(
                        context, nodeSchemaVersion, valueCount, ranks, dimensions,
                        nodeCount, nodes, feedCount, feeds, targetCount, targets, output);
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(EXECUTABLE_CREATE + " invocation failed", failure);
            }
        }

        private static int invokeRunExecutable(MethodHandle handle, MemorySegment executable,
                int inputCount, MemorySegment inputs, int outputCount, MemorySegment outputs) {
            try {
                return (int) handle.invokeExact(
                        executable, inputCount, inputs, outputCount, outputs);
            } catch (RuntimeException | Error failure) {
                throw failure;
            } catch (Throwable failure) {
                throw new IllegalStateException(EXECUTABLE_RUN + " invocation failed", failure);
            }
        }

        private static void closeAfterFailure(Arena arena, Throwable primary) {
            try {
                arena.close();
            } catch (RuntimeException | Error cleanup) {
                if (cleanup != primary) {
                    primary.addSuppressed(cleanup);
                }
            }
        }
    }
}
