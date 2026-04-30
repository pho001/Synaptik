package backend.cuda.bridge;

import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagSpec;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

/**
 * CUDA graph bridge backed by the Java Foreign Function and Memory API.
 *
 * <p>Availability is discovered once at class initialization by looking up the
 * native shim symbols. Missing libraries, missing ABI functions, and native
 * failures are reported through unavailable context/executable records so callers
 * can fall back to CPU execution.</p>
 */
public final class CudaFfmBridge implements CudaGraphBridge {
    private static final boolean CUDA_BUFFER_EXECUTION_ENABLED = false;
    private static final State STATE = init();
    private static volatile CudaBridgeContext SHARED_CONTEXT;

    /**
     * Returns whether the CUDA shim symbols required for bridge discovery were found.
     */
    @Override
    public boolean isAvailable() {
        return STATE.available;
    }

    /**
     * Returns the bridge discovery failure reason, or an empty string when available.
     */
    @Override
    public String unavailableReason() {
        return STATE.reason;
    }

    /**
     * Returns layered CUDA native bridge capability state.
     */
    @Override
    public CudaBridgeCapabilities capabilities() {
        return STATE.capabilities;
    }

    /**
     * Returns whether CUDA native buffer execution symbols and Java support are available.
     */
    @Override
    public boolean supportsBufferBindings() {
        return STATE.available
                && CUDA_BUFFER_EXECUTION_ENABLED
                && STATE.createBufferFn != null
                && STATE.destroyBufferFn != null
                && STATE.executePartitionBuffersFn != null;
    }

    /**
     * Creates or returns the shared CUDA context; failures return an unavailable context.
     */
    @Override
    public CudaBridgeContext createContext() {
        CudaBridgeContext cached = SHARED_CONTEXT;
        if (cached != null) {
            return cached;
        }
        if (!STATE.available) {
            return CudaBridgeContext.unavailable(unavailableReason());
        }
        if (STATE.createContextFn == null) {
            return CudaBridgeContext.unavailable("CUDA shim is available, but create_context ABI is missing.");
        }
        try {
            MemorySegment handle = (MemorySegment) STATE.createContextFn.invokeExact();
            if (handle == null || handle.equals(MemorySegment.NULL)) {
                return CudaBridgeContext.unavailable("CUDA shim returned a null context handle.");
            }
            CudaBridgeContext created = new CudaBridgeContext(true, handle, "");
            SHARED_CONTEXT = created;
            return created;
        } catch (Throwable t) {
            return CudaBridgeContext.unavailable("create_context failed: " + safeMessage(t));
        }
    }

    /**
     * Compiles a lowered accelerator DAG through the CUDA shim.
     */
    @Override
    public CudaBridgeExecutable compile(CudaBridgeContext bridgeContext, AcceleratorDagSpec dagSpec) {
        if (!STATE.available) {
            return CudaBridgeExecutable.unavailable(unavailableReason());
        }
        if (bridgeContext == null || !bridgeContext.available()) {
            return CudaBridgeExecutable.unavailable(bridgeContext == null ? "Missing CUDA bridge context." : bridgeContext.reason());
        }
        if (dagSpec == null) {
            return CudaBridgeExecutable.unavailable("CUDA FFM bridge requires lowered DAG spec.");
        }
        if (STATE.compilePartitionFn == null) {
            return CudaBridgeExecutable.unavailable("CUDA shim is available, but compile_partition ABI is missing.");
        }
        try (Arena compileArena = Arena.ofConfined()) {
            int externalInputCount = dagSpec.externalInputs().size();
            int[] externalInputRanks = new int[externalInputCount];
            int[] externalInputDTypes = new int[externalInputCount];
            int[] externalInputDim0 = new int[externalInputCount];
            int[] externalInputDim1 = new int[externalInputCount];
            int[] externalInputDim2 = new int[externalInputCount];
            int[] externalInputDim3 = new int[externalInputCount];
            for (int i = 0; i < externalInputCount; i++) {
                AcceleratorDagInput input = dagSpec.externalInputs().get(i);
                externalInputRanks[i] = input.shape().size();
                externalInputDTypes[i] = cudaDataTypeCode(input.dataType());
                externalInputDim0[i] = input.shape().getFirst();
                externalInputDim1[i] = input.shape().size() >= 2 ? input.shape().get(1) : 1;
                externalInputDim2[i] = input.shape().size() >= 3 ? input.shape().get(2) : 1;
                externalInputDim3[i] = input.shape().size() >= 4 ? input.shape().get(3) : 1;
            }
            int nodeCount = dagSpec.nodes().size();
            int[] nodeTypes = new int[nodeCount];
            int[] input0Kinds = new int[nodeCount];
            int[] input0Indices = new int[nodeCount];
            int[] input1Kinds = new int[nodeCount];
            int[] input1Indices = new int[nodeCount];
            int[] input2Kinds = new int[nodeCount];
            int[] input2Indices = new int[nodeCount];
            int[] input3Kinds = new int[nodeCount];
            int[] input3Indices = new int[nodeCount];
            float[] scalarValues = new float[nodeCount];
            int[] outputRanks = new int[nodeCount];
            int[] outputDim0 = new int[nodeCount];
            int[] outputDim1 = new int[nodeCount];
            int[] outputDim2 = new int[nodeCount];
            int[] outputDim3 = new int[nodeCount];
            for (int i = 0; i < nodeCount; i++) {
                AcceleratorDagNode node = dagSpec.nodes().get(i);
                nodeTypes[i] = node.type().abiCode();
                input0Kinds[i] = node.input0().kind().abiCode();
                input0Indices[i] = node.input0().index();
                input1Kinds[i] = node.input1().kind().abiCode();
                input1Indices[i] = node.input1().index();
                input2Kinds[i] = node.input2().kind().abiCode();
                input2Indices[i] = node.input2().index();
                input3Kinds[i] = node.input3().kind().abiCode();
                input3Indices[i] = node.input3().index();
                scalarValues[i] = Float.intBitsToFloat(node.scalarValueBits());
                outputRanks[i] = node.outputRank();
                outputDim0[i] = node.outputDim0();
                outputDim1[i] = node.outputDim1();
                outputDim2[i] = node.outputDim2();
                outputDim3[i] = node.outputDim3();
            }
            int outputCount = dagSpec.outputNodeIndices().size();
            int[] outputNodeIndices = dagSpec.outputNodeIndices().stream().mapToInt(Integer::intValue).toArray();
            MemorySegment handle = (MemorySegment) STATE.compilePartitionFn.invokeExact(
                    bridgeContext.handle(),
                    externalInputCount,
                    intArrayOrNull(compileArena, externalInputRanks),
                    intArrayOrNull(compileArena, externalInputDTypes),
                    intArrayOrNull(compileArena, externalInputDim0),
                    intArrayOrNull(compileArena, externalInputDim1),
                    intArrayOrNull(compileArena, externalInputDim2),
                    intArrayOrNull(compileArena, externalInputDim3),
                    nodeCount,
                    intArrayOrNull(compileArena, nodeTypes),
                    intArrayOrNull(compileArena, input0Kinds),
                    intArrayOrNull(compileArena, input0Indices),
                    intArrayOrNull(compileArena, input1Kinds),
                    intArrayOrNull(compileArena, input1Indices),
                    intArrayOrNull(compileArena, input2Kinds),
                    intArrayOrNull(compileArena, input2Indices),
                    intArrayOrNull(compileArena, input3Kinds),
                    intArrayOrNull(compileArena, input3Indices),
                    floatArrayOrNull(compileArena, scalarValues),
                    intArrayOrNull(compileArena, outputRanks),
                    intArrayOrNull(compileArena, outputDim0),
                    intArrayOrNull(compileArena, outputDim1),
                    intArrayOrNull(compileArena, outputDim2),
                    intArrayOrNull(compileArena, outputDim3),
                    outputCount,
                    intArrayOrNull(compileArena, outputNodeIndices)
            );
            if (handle == null || handle.equals(MemorySegment.NULL)) {
                return CudaBridgeExecutable.unavailable("CUDA shim returned a null executable handle.");
            }
            return new CudaBridgeExecutable(
                    true,
                    handle,
                    "",
                    false,
                    dagSpec.externalInputs().stream().map(AcceleratorDagInput::nodeId).toList(),
                    dagSpec.externalInputs().stream().map(AcceleratorDagInput::dataType).toList(),
                    dagSpec.outputNodeIds(),
                    dagSpec.outputNodeIds().stream().map(ignored -> DataType.FLOAT32).toList()
            );
        } catch (Throwable t) {
            return CudaBridgeExecutable.unavailable("compile_partition failed: " + safeMessage(t));
        }
    }

    /**
     * Executes a compiled CUDA DAG and copies native output buffers back into runtime tensors.
     */
    @Override
    public void execute(
            CudaBridgeContext bridgeContext,
            CudaBridgeExecutable executable,
            java.util.List<Tensor> externalInputs,
            java.util.List<Tensor> outputs
    ) {
        if (!STATE.available) {
            throw new UnsupportedOperationException(unavailableReason());
        }
        if (bridgeContext == null || !bridgeContext.available()) {
            throw new UnsupportedOperationException(bridgeContext == null ? "Missing CUDA bridge context." : bridgeContext.reason());
        }
        if (executable == null || !executable.available()) {
            throw new UnsupportedOperationException(executable == null ? "Missing CUDA bridge executable." : executable.reason());
        }
        if (STATE.executePartitionFn == null) {
            throw new UnsupportedOperationException("CUDA shim is available, but execute_partition ABI is missing.");
        }
        java.util.List<Tensor> resolvedExternalInputs = externalInputs == null ? java.util.List.of() : java.util.List.copyOf(externalInputs);
        java.util.List<Tensor> resolvedOutputs = outputs == null ? java.util.List.of() : java.util.List.copyOf(outputs);
        if (resolvedOutputs.isEmpty()) {
            throw new UnsupportedOperationException("CUDA FFM bridge requires at least one output tensor.");
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment externalInputSegs = resolvedExternalInputs.isEmpty() ? MemorySegment.NULL : arena.allocate(ADDRESS, resolvedExternalInputs.size());
            for (int i = 0; i < resolvedExternalInputs.size(); i++) {
                Tensor tensor = resolvedExternalInputs.get(i);
                if (tensor == null) {
                    throw new UnsupportedOperationException("CUDA FFM bridge received null external input.");
                }
                MemorySegment dataSeg;
                if (tensor.getDataType() == DataType.FLOAT32 && tensor.getFloat32Data() != null) {
                    dataSeg = arena.allocateFrom(JAVA_FLOAT, tensor.getFloat32Data());
                } else if (tensor.getDataType() == DataType.BOOL && tensor.getBoolData() != null) {
                    dataSeg = arena.allocateFrom(JAVA_BYTE, tensor.getBoolData());
                } else {
                    throw new UnsupportedOperationException("CUDA FFM bridge currently supports only FLOAT32/BOOL external inputs.");
                }
                externalInputSegs.setAtIndex(ADDRESS, i, dataSeg);
            }
            MemorySegment outputSegs = arena.allocate(ADDRESS, resolvedOutputs.size());
            for (int i = 0; i < resolvedOutputs.size(); i++) {
                Tensor out = resolvedOutputs.get(i);
                float[] outData = out == null ? null : out.getFloat32Data();
                if (outData == null) {
                    throw new UnsupportedOperationException("CUDA FFM bridge currently supports only FLOAT32 output tensors with direct float[] storage.");
                }
                outputSegs.setAtIndex(ADDRESS, i, arena.allocate(JAVA_FLOAT, outData.length));
            }
            int status = (int) STATE.executePartitionFn.invokeExact(
                    bridgeContext.handle(),
                    executable.handle(),
                    externalInputSegs,
                    resolvedExternalInputs.size(),
                    outputSegs,
                    resolvedOutputs.size()
            );
            if (status == 0) {
                for (int i = 0; i < resolvedOutputs.size(); i++) {
                    Tensor out = resolvedOutputs.get(i);
                    float[] outData = out.getFloat32Data();
                    MemorySegment.ofArray(outData).copyFrom(outputSegs.getAtIndex(ADDRESS, i).reinterpret((long) outData.length * JAVA_FLOAT.byteSize()));
                    out.markDataViewStale();
                }
                return;
            }
            throw new UnsupportedOperationException("CUDA execute_partition returned non-zero status: " + status);
        } catch (Throwable t) {
            throw new UnsupportedOperationException("CUDA execute_partition failed: " + safeMessage(t), t);
        }
    }

    /**
     * Releases a non-shared native context when the shim exposes a destroy function.
     */
    @Override
    public void destroyContext(CudaBridgeContext bridgeContext) {
        if (bridgeContext == SHARED_CONTEXT) {
            return;
        }
        if (!STATE.available || STATE.destroyContextFn == null || bridgeContext == null || !bridgeContext.available()) {
            return;
        }
        try {
            STATE.destroyContextFn.invokeExact(bridgeContext.handle());
        } catch (Throwable ignored) {
        }
    }

    /**
     * Releases a native executable when the shim exposes a destroy function.
     */
    @Override
    public void destroyExecutable(CudaBridgeExecutable executable) {
        if (!STATE.available || STATE.destroyExecutableFn == null || executable == null || !executable.available()) {
            return;
        }
        try {
            STATE.destroyExecutableFn.invokeExact(executable.handle());
        } catch (Throwable ignored) {
        }
    }

    private static MemorySegment intArrayOrNull(Arena arena, int[] values) {
        return values == null || values.length == 0 ? MemorySegment.NULL : arena.allocateFrom(JAVA_INT, values);
    }

    private static MemorySegment floatArrayOrNull(Arena arena, float[] values) {
        return values == null || values.length == 0 ? MemorySegment.NULL : arena.allocateFrom(JAVA_FLOAT, values);
    }

    private static int cudaDataTypeCode(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> 1;
            case BOOL -> 2;
            default -> throw new IllegalArgumentException("CUDA DAG bridge currently supports FLOAT32/BOOL tensors, got " + dataType);
        };
    }

    private static State init() {
        Arena arena = Arena.ofShared();
        SymbolLookup lookup;
        try {
            lookup = resolveLookup(arena);
        } catch (Throwable t) {
            String reason = t.getClass().getSimpleName() + ": " + safeMessage(t);
            return State.unavailable(CudaBridgeCapabilityCode.NATIVE_LIBRARY_UNAVAILABLE, reason, arena);
        }
        try {
            Linker linker = Linker.nativeLinker();

            MemorySegment availableSymbol = lookup.find("synaptik_cuda_graph_available").orElse(null);
            MemorySegment unavailableReasonSymbol = lookup.find("synaptik_cuda_graph_unavailable_reason").orElse(null);
            if (availableSymbol == null || unavailableReasonSymbol == null) {
                String missing = availableSymbol == null
                        ? "synaptik_cuda_graph_available"
                        : "synaptik_cuda_graph_unavailable_reason";
                return State.unavailable(
                        CudaBridgeCapabilityCode.REQUIRED_SYMBOL_MISSING,
                        "CUDA shim is missing required symbol: " + missing,
                        arena
                );
            }
            MethodHandle availableFn = linker.downcallHandle(
                    availableSymbol,
                    FunctionDescriptor.of(JAVA_INT)
            );
            MethodHandle unavailableReasonFn = linker.downcallHandle(
                    unavailableReasonSymbol,
                    FunctionDescriptor.of(ADDRESS)
            );
            MethodHandle createContextFn = optionalHandle(linker, lookup, "synaptik_cuda_graph_create_context", FunctionDescriptor.of(ADDRESS));
            MethodHandle compilePartitionFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_cuda_graph_compile_partition_f32",
                    FunctionDescriptor.of(
                            ADDRESS,
                            ADDRESS,
                            JAVA_INT,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            JAVA_INT,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            JAVA_INT,
                            ADDRESS
                    )
            );
            MethodHandle executePartitionFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_cuda_graph_execute_partition_f32",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT)
            );
            MethodHandle destroyContextFn = optionalHandle(linker, lookup, "synaptik_cuda_graph_destroy_context", FunctionDescriptor.ofVoid(ADDRESS));
            MethodHandle destroyExecutableFn = optionalHandle(linker, lookup, "synaptik_cuda_graph_destroy_executable", FunctionDescriptor.ofVoid(ADDRESS));
            MethodHandle createBufferFn = optionalHandle(linker, lookup, "synaptik_cuda_graph_create_buffer", FunctionDescriptor.of(ADDRESS, ADDRESS, ADDRESS, JAVA_INT));
            MethodHandle destroyBufferFn = optionalHandle(linker, lookup, "synaptik_cuda_graph_destroy_buffer", FunctionDescriptor.ofVoid(ADDRESS));
            MethodHandle executePartitionBuffersFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_cuda_graph_execute_partition_f32_buffers",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT)
            );

            int available = (int) availableFn.invokeExact();
            if (available == 0) {
                MemorySegment reasonPtr = (MemorySegment) unavailableReasonFn.invokeExact();
                String reason = cStringOrDefault(reasonPtr, "CUDA shim reported unavailable.");
                return new State(
                        false,
                        reason,
                        arena,
                        createContextFn,
                        compilePartitionFn,
                        executePartitionFn,
                        destroyContextFn,
                        destroyExecutableFn,
                        createBufferFn,
                        destroyBufferFn,
                        executePartitionBuffersFn,
                        CudaBridgeCapabilities.unavailable(CudaBridgeCapabilityCode.CUDA_RUNTIME_UNAVAILABLE, reason)
                );
            }
            if (createContextFn == null || compilePartitionFn == null || executePartitionFn == null) {
                String missing = createContextFn == null
                        ? "synaptik_cuda_graph_create_context"
                        : compilePartitionFn == null
                        ? "synaptik_cuda_graph_compile_partition_f32"
                        : "synaptik_cuda_graph_execute_partition_f32";
                String reason = "CUDA shim is missing graph execution ABI symbol: " + missing;
                return new State(
                        false,
                        reason,
                        arena,
                        createContextFn,
                        compilePartitionFn,
                        executePartitionFn,
                        destroyContextFn,
                        destroyExecutableFn,
                        createBufferFn,
                        destroyBufferFn,
                        executePartitionBuffersFn,
                        CudaBridgeCapabilities.unavailable(CudaBridgeCapabilityCode.GRAPH_EXECUTION_ABI_UNAVAILABLE, reason)
                );
            }
            boolean bufferSupported = CUDA_BUFFER_EXECUTION_ENABLED
                    && createBufferFn != null
                    && destroyBufferFn != null
                    && executePartitionBuffersFn != null;
            return new State(
                    true,
                    "",
                    arena,
                    createContextFn,
                    compilePartitionFn,
                    executePartitionFn,
                    destroyContextFn,
                    destroyExecutableFn,
                    createBufferFn,
                    destroyBufferFn,
                    executePartitionBuffersFn,
                    CudaBridgeCapabilities.available(bufferSupported)
            );
        } catch (Throwable t) {
            String reason = t.getClass().getSimpleName() + ": " + safeMessage(t);
            return State.unavailable(CudaBridgeCapabilityCode.REQUIRED_SYMBOL_MISSING, reason, arena);
        }
    }

    private static MethodHandle optionalHandle(
            Linker linker,
            SymbolLookup lookup,
            String symbol,
            FunctionDescriptor descriptor
    ) {
        MemorySegment segment = lookup.find(symbol).orElse(null);
        return segment == null ? null : linker.downcallHandle(segment, descriptor);
    }

    private static SymbolLookup resolveLookup(Arena arena) {
        String explicit = System.getProperty("synaptik.cuda.graph.lib");
        if (explicit != null && !explicit.isBlank()) {
            return SymbolLookup.libraryLookup(explicit.trim(), arena);
        }
        String envLib = System.getenv("SYNAPTIK_CUDA_GRAPH_LIB");
        if (envLib != null && !envLib.isBlank()) {
            return SymbolLookup.libraryLookup(envLib.trim(), arena);
        }
        return SymbolLookup.libraryLookup("synaptik_cuda_graph", arena);
    }

    private static String cStringOrDefault(MemorySegment ptr, String fallback) {
        if (ptr == null || ptr.equals(MemorySegment.NULL)) {
            return fallback;
        }
        try {
            return ptr.reinterpret(Long.MAX_VALUE).getString(0, StandardCharsets.UTF_8);
        } catch (Throwable ignored) {
            return fallback;
        }
    }

    private static String safeMessage(Throwable t) {
        String message = t.getMessage();
        return message == null || message.isBlank() ? "<no-message>" : message;
    }

    private static final class State {
        private final boolean available;
        private final String reason;
        @SuppressWarnings("unused")
        private final Arena arenaRef;
        private final MethodHandle createContextFn;
        private final MethodHandle compilePartitionFn;
        private final MethodHandle executePartitionFn;
        private final MethodHandle destroyContextFn;
        private final MethodHandle destroyExecutableFn;
        private final MethodHandle createBufferFn;
        private final MethodHandle destroyBufferFn;
        private final MethodHandle executePartitionBuffersFn;
        private final CudaBridgeCapabilities capabilities;

        private State(
                boolean available,
                String reason,
                Arena arenaRef,
                MethodHandle createContextFn,
                MethodHandle compilePartitionFn,
                MethodHandle executePartitionFn,
                MethodHandle destroyContextFn,
                MethodHandle destroyExecutableFn,
                MethodHandle createBufferFn,
                MethodHandle destroyBufferFn,
                MethodHandle executePartitionBuffersFn,
                CudaBridgeCapabilities capabilities
        ) {
            this.available = available;
            this.reason = reason == null ? "" : reason;
            this.arenaRef = arenaRef;
            this.createContextFn = createContextFn;
            this.compilePartitionFn = compilePartitionFn;
            this.executePartitionFn = executePartitionFn;
            this.destroyContextFn = destroyContextFn;
            this.destroyExecutableFn = destroyExecutableFn;
            this.createBufferFn = createBufferFn;
            this.destroyBufferFn = destroyBufferFn;
            this.executePartitionBuffersFn = executePartitionBuffersFn;
            this.capabilities = capabilities == null
                    ? CudaBridgeCapabilities.unavailable(CudaBridgeCapabilityCode.NATIVE_LIBRARY_UNAVAILABLE, this.reason)
                    : capabilities;
        }

        private static State unavailable(CudaBridgeCapabilityCode code, String reason, Arena arenaRef) {
            return new State(
                    false,
                    reason,
                    arenaRef,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    null,
                    CudaBridgeCapabilities.unavailable(code, reason)
            );
        }
    }
}
