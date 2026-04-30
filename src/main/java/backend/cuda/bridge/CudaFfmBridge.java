package backend.cuda.bridge;

import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.buffer.AcceleratorLayoutAbiV2Descriptor;
import backend.accelerator.buffer.AcceleratorLayoutAbiV2Support;
import backend.cuda.buffer.CudaBufferAllocator;
import backend.cuda.buffer.CudaBufferBinding;
import backend.cuda.buffer.CudaBufferHandle;
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
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * CUDA graph bridge backed by the Java Foreign Function and Memory API.
 *
 * <p>Availability is discovered once at class initialization by looking up the
 * native shim symbols. Missing libraries, missing ABI functions, and native
 * failures are reported through unavailable context/executable records so callers
 * can fall back to CPU execution.</p>
 */
public final class CudaFfmBridge implements CudaGraphBridge {
    private static final boolean CUDA_BUFFER_EXECUTION_ENABLED = true;
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
                && STATE.readBufferFn != null
                && STATE.destroyBufferFn != null
                && STATE.executePartitionBuffersFn != null;
    }

    @Override
    public boolean supportsLayoutMaterialization() {
        return STATE.available && STATE.layoutContiguousF32BufferFn != null;
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

    /**
     * Creates a run-scoped CUDA buffer allocator backed by native FFM calls.
     */
    @Override
    public CudaBufferAllocator createBufferAllocator(CudaBridgeContext bridgeContext) {
        if (!STATE.available || bridgeContext == null || !bridgeContext.available()) {
            return CudaBufferAllocator.unavailable("CUDA bridge context is unavailable.");
        }
        if (!supportsBufferBindings()) {
            return CudaBufferAllocator.unavailable("native CUDA buffer ABI unavailable: bridge does not support buffer bindings");
        }
        return CudaBufferAllocator.available(new CudaBufferAllocator.NativeAccess() {
            @Override
            public CudaBufferHandle createBuffer(long byteLength, MemorySegment initialData, long initialDataBytes) {
                try {
                    int bytes = checkedInt(byteLength, "byteLength");
                    MemorySegment data = initialData == null ? MemorySegment.NULL : initialData;
                    MemorySegment handle = (MemorySegment) STATE.createBufferFn.invokeExact(
                            bridgeContext.handle(),
                            data,
                            bytes
                    );
                    if (handle == null || handle.equals(MemorySegment.NULL)) {
                        throw new UnsupportedOperationException("CUDA create_buffer returned a null handle.");
                    }
                    return new CudaBufferHandle(handle, byteLength, true);
                } catch (Throwable t) {
                    throw new UnsupportedOperationException("CUDA create_buffer failed: " + safeMessage(t), t);
                }
            }

            @Override
            public void readBuffer(CudaBufferHandle handle, MemorySegment destination, long byteLength) {
                if (handle == null || !handle.available()) {
                    throw new UnsupportedOperationException("CUDA read_buffer requires an available handle.");
                }
                try {
                    int status = (int) STATE.readBufferFn.invokeExact(
                            bridgeContext.handle(),
                            handle.handle(),
                            destination,
                            checkedInt(byteLength, "byteLength")
                    );
                    if (status != 0) {
                        throw new UnsupportedOperationException("CUDA read_buffer returned non-zero status: " + status);
                    }
                } catch (Throwable t) {
                    throw new UnsupportedOperationException("CUDA read_buffer failed: " + safeMessage(t), t);
                }
            }

            @Override
            public void destroyBuffer(CudaBufferHandle handle) {
                if (handle == null || !handle.available()) {
                    return;
                }
                try {
                    STATE.destroyBufferFn.invokeExact(handle.handle());
                } catch (Throwable ignored) {
                }
            }
        });
    }

    @Override
    public void materializeLayout(
            CudaBridgeContext context,
            CudaBufferBinding source,
            CudaBufferBinding destination
    ) {
        if (!supportsLayoutMaterialization()) {
            throw new UnsupportedOperationException("CUDA layout materialization symbol is unavailable.");
        }
        if (context == null || !context.available()) {
            throw new UnsupportedOperationException(context == null ? "Missing CUDA bridge context." : context.reason());
        }
        if (source == null || !source.available() || destination == null || !destination.available()) {
            throw new UnsupportedOperationException("CUDA layout materialization requires available source and destination bindings.");
        }
        AcceleratorLayoutAbiV2Descriptor descriptor = AcceleratorLayoutAbiV2Descriptor.fromBinding(source);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shape = longArray(arena, descriptor.shape());
            MemorySegment strides = longArray(arena, descriptor.strides());
            int status = (int) STATE.layoutContiguousF32BufferFn.invokeExact(
                    context.handle(),
                    source.handle().handle(),
                    destination.handle().handle(),
                    descriptor.rank(),
                    shape,
                    strides,
                    (long) descriptor.storageOffset(),
                    descriptor.logicalElementCount(),
                    descriptor.physicalByteSpan(),
                    destination.logicalByteLength()
            );
            if (status != 0) {
                throw new UnsupportedOperationException("CUDA layout materialization returned non-zero status: " + status);
            }
        } catch (Throwable t) {
            throw new UnsupportedOperationException("CUDA layout materialization failed: " + safeMessage(t), t);
        }
    }

    /**
     * Executes a compiled CUDA DAG using native buffer handles.
     */
    @Override
    public void executeBuffers(
            CudaBridgeContext bridgeContext,
            CudaBridgeExecutable executable,
            java.util.List<CudaBufferBinding> inputs,
            java.util.List<CudaBufferBinding> outputs
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
        if (!supportsBufferBindings()) {
            throw new UnsupportedOperationException("native CUDA buffer ABI unavailable: bridge does not support buffer bindings");
        }
        java.util.List<CudaBufferBinding> resolvedInputs = inputs == null ? java.util.List.of() : java.util.List.copyOf(inputs);
        java.util.List<CudaBufferBinding> resolvedOutputs = outputs == null ? java.util.List.of() : java.util.List.copyOf(outputs);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment inputHandles = bufferHandleArray(arena, resolvedInputs);
            MemorySegment outputHandles = bufferHandleArray(arena, resolvedOutputs);
            int status = (int) STATE.executePartitionBuffersFn.invokeExact(
                    bridgeContext.handle(),
                    executable.handle(),
                    inputHandles,
                    resolvedInputs.size(),
                    outputHandles,
                    resolvedOutputs.size()
            );
            if (status != 0) {
                throw new UnsupportedOperationException("CUDA execute_partition_f32_buffers returned non-zero status: " + status);
            }
        } catch (Throwable t) {
            throw new UnsupportedOperationException("CUDA execute_partition_f32_buffers failed: " + safeMessage(t), t);
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
                        arena,
                        true
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
            MethodHandle readBufferFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_cuda_graph_read_buffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT)
            );
            MethodHandle destroyBufferFn = optionalHandle(linker, lookup, "synaptik_cuda_graph_destroy_buffer", FunctionDescriptor.ofVoid(ADDRESS));
            MethodHandle executePartitionBuffersFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_cuda_graph_execute_partition_f32_buffers",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT)
            );
            MethodHandle layoutAbiVersionFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_cuda_graph_layout_abi_version",
                    FunctionDescriptor.of(JAVA_INT)
            );
            MethodHandle validateLayoutAbiV2Fn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_cuda_graph_validate_layout_abi_v2",
                    FunctionDescriptor.of(
                            JAVA_INT,
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
                            ADDRESS
                    )
            );
            MethodHandle layoutContiguousF32BufferFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_cuda_graph_layout_contiguous_f32_buffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG)
            );
            int layoutAbiV2Version = layoutAbiVersion(layoutAbiVersionFn);
            boolean layoutAbiV2Supported = layoutAbiV2Version == AcceleratorLayoutAbiV2Support.REQUIRED_VERSION
                    && validateLayoutAbiV2Fn != null;

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
                        readBufferFn,
                        destroyBufferFn,
                        executePartitionBuffersFn,
                        layoutContiguousF32BufferFn,
                        new CudaBridgeCapabilities(
                                true,
                                false,
                                false,
                                false,
                                false,
                                layoutAbiV2Supported,
                                layoutAbiV2Version,
                                CudaBridgeCapabilityCode.CUDA_RUNTIME_UNAVAILABLE,
                                reason
                        )
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
                        readBufferFn,
                        destroyBufferFn,
                        executePartitionBuffersFn,
                        layoutContiguousF32BufferFn,
                        new CudaBridgeCapabilities(
                                true,
                                true,
                                createContextFn != null,
                                false,
                                false,
                                layoutAbiV2Supported,
                                layoutAbiV2Version,
                                CudaBridgeCapabilityCode.GRAPH_EXECUTION_ABI_UNAVAILABLE,
                                reason
                        )
                );
            }
            boolean bufferSupported = CUDA_BUFFER_EXECUTION_ENABLED
                    && createBufferFn != null
                    && readBufferFn != null
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
                    readBufferFn,
                    destroyBufferFn,
                    executePartitionBuffersFn,
                    layoutContiguousF32BufferFn,
                    new CudaBridgeCapabilities(
                            true,
                            true,
                            true,
                            true,
                            bufferSupported,
                            layoutAbiV2Supported,
                            layoutAbiV2Version,
                            cudaCapabilityCode(layoutAbiV2Version, layoutAbiV2Supported),
                            cudaCapabilityReason(layoutAbiV2Version, layoutAbiV2Supported)
                    )
            );
        } catch (Throwable t) {
            String reason = t.getClass().getSimpleName() + ": " + safeMessage(t);
            return State.unavailable(CudaBridgeCapabilityCode.REQUIRED_SYMBOL_MISSING, reason, arena, true);
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

    private static MemorySegment longArray(Arena arena, int[] values) {
        long[] out = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i];
        }
        return out.length == 0 ? MemorySegment.NULL : arena.allocateFrom(JAVA_LONG, out);
    }

    private static int layoutAbiVersion(MethodHandle layoutAbiVersionFn) {
        if (layoutAbiVersionFn == null) {
            return 0;
        }
        try {
            return Math.max(0, (int) layoutAbiVersionFn.invokeExact());
        } catch (Throwable ignored) {
            return 0;
        }
    }

    private static CudaBridgeCapabilityCode cudaCapabilityCode(int layoutAbiV2Version, boolean layoutAbiV2Supported) {
        if (layoutAbiV2Supported) {
            return CudaBridgeCapabilityCode.AVAILABLE;
        }
        if (layoutAbiV2Version == 0) {
            return CudaBridgeCapabilityCode.LAYOUT_ABI_V2_UNAVAILABLE;
        }
        return CudaBridgeCapabilityCode.LAYOUT_ABI_V2_VERSION_MISMATCH;
    }

    private static String cudaCapabilityReason(int layoutAbiV2Version, boolean layoutAbiV2Supported) {
        if (layoutAbiV2Supported) {
            return "";
        }
        if (layoutAbiV2Version == 0) {
            return "CUDA layout ABI v2 symbols unavailable";
        }
        return "CUDA layout ABI v2 version mismatch: expected "
                + AcceleratorLayoutAbiV2Support.REQUIRED_VERSION
                + ", got "
                + layoutAbiV2Version;
    }

    private static MemorySegment bufferHandleArray(Arena arena, java.util.List<CudaBufferBinding> bindings) {
        if (bindings == null || bindings.isEmpty()) {
            return MemorySegment.NULL;
        }
        MemorySegment handles = arena.allocate(ADDRESS, bindings.size());
        for (int i = 0; i < bindings.size(); i++) {
            CudaBufferBinding binding = bindings.get(i);
            if (binding == null || !binding.available()) {
                throw new UnsupportedOperationException("CUDA buffer binding " + i + " is unavailable.");
            }
            handles.setAtIndex(ADDRESS, i, binding.handle().handle());
        }
        return handles;
    }

    private static int checkedInt(long value, String label) {
        if (value < 0L || value > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(label + " out of int range: " + value);
        }
        return (int) value;
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
        private final MethodHandle readBufferFn;
        private final MethodHandle destroyBufferFn;
        private final MethodHandle executePartitionBuffersFn;
        private final MethodHandle layoutContiguousF32BufferFn;
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
                MethodHandle readBufferFn,
                MethodHandle destroyBufferFn,
                MethodHandle executePartitionBuffersFn,
                MethodHandle layoutContiguousF32BufferFn,
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
            this.readBufferFn = readBufferFn;
            this.destroyBufferFn = destroyBufferFn;
            this.executePartitionBuffersFn = executePartitionBuffersFn;
            this.layoutContiguousF32BufferFn = layoutContiguousF32BufferFn;
            this.capabilities = capabilities == null
                    ? CudaBridgeCapabilities.unavailable(CudaBridgeCapabilityCode.NATIVE_LIBRARY_UNAVAILABLE, this.reason)
                    : capabilities;
        }

        private static State unavailable(CudaBridgeCapabilityCode code, String reason, Arena arenaRef) {
            return unavailable(code, reason, arenaRef, false);
        }

        private static State unavailable(
                CudaBridgeCapabilityCode code,
                String reason,
                Arena arenaRef,
                boolean nativeLibraryAvailable
        ) {
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
                    null,
                    null,
                    new CudaBridgeCapabilities(
                            nativeLibraryAvailable,
                            false,
                            false,
                            false,
                            false,
                            false,
                            0,
                            code,
                            reason
                    )
            );
        }
    }
}
