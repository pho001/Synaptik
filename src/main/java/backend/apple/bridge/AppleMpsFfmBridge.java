package backend.apple.bridge;

import backend.apple.lowering.AppleGpuPartitionPlan;
import backend.apple.lowering.AppleGpuSubgraphSignature;
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
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;

public final class AppleMpsFfmBridge implements AppleMpsGraphBridge {
    private static final State STATE = init();
    private static final ConcurrentMap<AppleGpuSubgraphSignature, MemorySegment> EXECUTABLE_CACHE = new ConcurrentHashMap<>();
    private static volatile AppleMpsBridgeContext SHARED_CONTEXT;

    @Override
    public boolean isAvailable() {
        return STATE.available;
    }

    @Override
    public String unavailableReason() {
        return STATE.reason;
    }

    @Override
    public AppleMpsBridgeContext createContext() {
        AppleMpsBridgeContext cached = SHARED_CONTEXT;
        if (cached != null) {
            return cached;
        }
        if (!STATE.available) {
            return AppleMpsBridgeContext.unavailable(unavailableReason());
        }
        if (STATE.createContextFn == null) {
            return AppleMpsBridgeContext.unavailable("Apple MPS shim is available, but create_context ABI is missing.");
        }
        try {
            MemorySegment handle = (MemorySegment) STATE.createContextFn.invokeExact();
            if (handle == null || handle.equals(MemorySegment.NULL)) {
                return AppleMpsBridgeContext.unavailable("Apple MPS shim returned a null context handle.");
            }
            AppleMpsBridgeContext created = new AppleMpsBridgeContext(true, handle, "");
            SHARED_CONTEXT = created;
            return created;
        } catch (Throwable t) {
            return AppleMpsBridgeContext.unavailable("create_context failed: " + safeMessage(t));
        }
    }

    @Override
    public AppleMpsBridgeExecutable compile(AppleMpsBridgeContext bridgeContext, AppleGpuPartitionPlan plan) {
        if (!STATE.available) {
            return AppleMpsBridgeExecutable.unavailable(unavailableReason());
        }
        if (bridgeContext == null || !bridgeContext.available()) {
            return AppleMpsBridgeExecutable.unavailable(bridgeContext == null ? "Missing Apple bridge context." : bridgeContext.reason());
        }
        if (plan == null || plan.lowering() == null || plan.lowering().dagSpec() == null) {
            return AppleMpsBridgeExecutable.unavailable("Apple FFM bridge requires lowered DAG spec.");
        }
        if (STATE.compilePartitionFn == null) {
            return AppleMpsBridgeExecutable.unavailable("Apple MPS shim is available, but compile_partition ABI is missing.");
        }
        try {
            AcceleratorDagSpec dagSpec = plan.lowering().dagSpec();
            AppleGpuSubgraphSignature signature = AppleGpuSubgraphSignature.from(plan);
            boolean cacheHit = EXECUTABLE_CACHE.containsKey(signature);
            MemorySegment handle = EXECUTABLE_CACHE.computeIfAbsent(signature, ignored -> {
                try {
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
                            externalInputDTypes[i] = appleDataTypeCode(input.dataType());
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
                        MemorySegment externalInputRanksSeg = externalInputRanks.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, externalInputRanks);
                        MemorySegment externalInputDTypesSeg = externalInputDTypes.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, externalInputDTypes);
                        MemorySegment externalInputDim0Seg = externalInputDim0.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, externalInputDim0);
                        MemorySegment externalInputDim1Seg = externalInputDim1.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, externalInputDim1);
                        MemorySegment externalInputDim2Seg = externalInputDim2.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, externalInputDim2);
                        MemorySegment externalInputDim3Seg = externalInputDim3.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, externalInputDim3);
                        MemorySegment nodeTypesSeg = nodeTypes.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, nodeTypes);
                        MemorySegment input0KindsSeg = input0Kinds.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, input0Kinds);
                        MemorySegment input0IndicesSeg = input0Indices.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, input0Indices);
                        MemorySegment input1KindsSeg = input1Kinds.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, input1Kinds);
                        MemorySegment input1IndicesSeg = input1Indices.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, input1Indices);
                        MemorySegment input2KindsSeg = input2Kinds.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, input2Kinds);
                        MemorySegment input2IndicesSeg = input2Indices.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, input2Indices);
                        MemorySegment input3KindsSeg = input3Kinds.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, input3Kinds);
                        MemorySegment input3IndicesSeg = input3Indices.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, input3Indices);
                        MemorySegment scalarValuesSeg = scalarValues.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_FLOAT, scalarValues);
                        MemorySegment outputRanksSeg = outputRanks.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, outputRanks);
                        MemorySegment outputDim0Seg = outputDim0.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, outputDim0);
                        MemorySegment outputDim1Seg = outputDim1.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, outputDim1);
                        MemorySegment outputDim2Seg = outputDim2.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, outputDim2);
                        MemorySegment outputDim3Seg = outputDim3.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, outputDim3);
                        int outputCount = dagSpec.outputNodeIndices().size();
                        int[] outputNodeIndices = dagSpec.outputNodeIndices().stream().mapToInt(Integer::intValue).toArray();
                        MemorySegment outputNodeIndicesSeg = outputNodeIndices.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, outputNodeIndices);
                        return (MemorySegment) STATE.compilePartitionFn.invokeExact(
                            bridgeContext.handle(),
                            externalInputCount,
                            externalInputRanksSeg,
                            externalInputDTypesSeg,
                            externalInputDim0Seg,
                            externalInputDim1Seg,
                            externalInputDim2Seg,
                            externalInputDim3Seg,
                            nodeCount,
                            nodeTypesSeg,
                            input0KindsSeg,
                            input0IndicesSeg,
                            input1KindsSeg,
                            input1IndicesSeg,
                            input2KindsSeg,
                            input2IndicesSeg,
                            input3KindsSeg,
                            input3IndicesSeg,
                            scalarValuesSeg,
                            outputRanksSeg,
                            outputDim0Seg,
                            outputDim1Seg,
                            outputDim2Seg,
                            outputDim3Seg,
                            outputCount,
                            outputNodeIndicesSeg
                        );
                    }
                } catch (Throwable t) {
                    return MemorySegment.NULL;
                }
            });
            if (handle == null || handle.equals(MemorySegment.NULL)) {
                return AppleMpsBridgeExecutable.unavailable("Apple MPS shim returned a null executable handle.");
            }
            return new AppleMpsBridgeExecutable(
                    true,
                    handle,
                    "",
                    cacheHit,
                    dagSpec.externalInputs().stream().map(AcceleratorDagInput::nodeId).toList(),
                    dagSpec.outputNodeIds(),
                    dagSpec.outputNodeIndices()
            );
        } catch (Throwable t) {
            return AppleMpsBridgeExecutable.unavailable("compile_partition failed: " + safeMessage(t));
        }
    }

    @Override
    public void destroyContext(AppleMpsBridgeContext bridgeContext) {
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

    @Override
    public void destroyExecutable(AppleMpsBridgeExecutable executable) {
        // Executables are cached by subgraph signature and reused within the shared bridge context.
    }

    @Override
    public void execute(
            AppleMpsBridgeContext bridgeContext,
            AppleMpsBridgeExecutable executable,
            java.util.List<Tensor> externalInputs,
            java.util.List<Tensor> outputs
    ) {
        if (!STATE.available) {
            throw new UnsupportedOperationException(unavailableReason());
        }
        if (bridgeContext == null || !bridgeContext.available()) {
            throw new UnsupportedOperationException(bridgeContext == null ? "Missing Apple bridge context." : bridgeContext.reason());
        }
        if (executable == null || !executable.available()) {
            throw new UnsupportedOperationException(executable == null ? "Missing Apple bridge executable." : executable.reason());
        }
        if (STATE.executePartitionFn == null) {
            throw new UnsupportedOperationException("Apple MPS shim is available, but execute_partition ABI is missing.");
        }
        java.util.List<Tensor> resolvedExternalInputs = externalInputs == null ? java.util.List.of() : java.util.List.copyOf(externalInputs);
        java.util.List<Tensor> resolvedOutputs = outputs == null ? java.util.List.of() : java.util.List.copyOf(outputs);
        if (resolvedOutputs.isEmpty()) {
            throw new UnsupportedOperationException("Apple MPS FFM bridge requires at least one output tensor.");
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment externalInputSegs = resolvedExternalInputs.isEmpty() ? MemorySegment.NULL : arena.allocate(ADDRESS, resolvedExternalInputs.size());
            for (int i = 0; i < resolvedExternalInputs.size(); i++) {
                Tensor tensor = resolvedExternalInputs.get(i);
                if (tensor == null) {
                    throw new UnsupportedOperationException("Apple MPS FFM bridge received null external input.");
                }
                MemorySegment dataSeg;
                if (tensor.getDataType() == DataType.FLOAT32 && tensor.getFloat32Data() != null) {
                    dataSeg = arena.allocateFrom(JAVA_FLOAT, tensor.getFloat32Data());
                } else if (tensor.getDataType() == DataType.BOOL && tensor.getBoolData() != null) {
                    dataSeg = arena.allocateFrom(JAVA_BYTE, tensor.getBoolData());
                } else {
                    throw new UnsupportedOperationException("Apple MPS FFM bridge currently supports only FLOAT32/BOOL external inputs.");
                }
                externalInputSegs.setAtIndex(ADDRESS, i, dataSeg);
            }
            MemorySegment outputSegs = arena.allocate(ADDRESS, resolvedOutputs.size());
            MemorySegment outputLengths = arena.allocate(JAVA_INT, resolvedOutputs.size());
            for (int i = 0; i < resolvedOutputs.size(); i++) {
                Tensor out = resolvedOutputs.get(i);
                float[] outData = out == null ? null : out.getFloat32Data();
                if (outData == null) {
                    throw new UnsupportedOperationException("Apple MPS FFM bridge currently supports only FLOAT32 output tensors with direct float[] storage.");
                }
                MemorySegment outSeg = arena.allocate(JAVA_FLOAT, outData.length);
                outputSegs.setAtIndex(ADDRESS, i, outSeg);
                outputLengths.setAtIndex(JAVA_INT, i, outData.length);
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
            throw new UnsupportedOperationException("Apple MPS execute_partition returned non-zero status: " + status);
        } catch (Throwable t) {
            throw new UnsupportedOperationException("Apple MPS execute_partition failed: " + safeMessage(t), t);
        }
    }

    private static State init() {
        try {
            Arena arena = Arena.ofShared();
            SymbolLookup lookup = resolveLookup(arena);
            Linker linker = Linker.nativeLinker();

            MethodHandle availableFn = linker.downcallHandle(
                    lookup.find("synaptik_apple_mps_available").orElseThrow(),
                    FunctionDescriptor.of(JAVA_INT)
            );
            MethodHandle unavailableReasonFn = linker.downcallHandle(
                    lookup.find("synaptik_apple_mps_unavailable_reason").orElseThrow(),
                    FunctionDescriptor.of(ADDRESS)
            );
            MethodHandle createContextFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_create_context",
                    FunctionDescriptor.of(ADDRESS)
            );
            MethodHandle compilePartitionFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_compile_partition_f32",
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
                    "synaptik_apple_mps_execute_partition_f32",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT)
            );
            MethodHandle destroyContextFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_destroy_context",
                    FunctionDescriptor.ofVoid(ADDRESS)
            );
            MethodHandle destroyExecutableFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_destroy_executable",
                    FunctionDescriptor.ofVoid(ADDRESS)
            );

            int available = (int) availableFn.invokeExact();
            if (available != 0) {
                return new State(true, null, arena, availableFn, unavailableReasonFn, createContextFn, compilePartitionFn, executePartitionFn, destroyContextFn, destroyExecutableFn);
            }

            MemorySegment reasonPtr = (MemorySegment) unavailableReasonFn.invokeExact();
            String reason = cStringOrDefault(reasonPtr, "Apple shim reported unavailable.");
            return new State(false, reason, arena, availableFn, unavailableReasonFn, createContextFn, compilePartitionFn, executePartitionFn, destroyContextFn, destroyExecutableFn);
        } catch (Throwable t) {
            return new State(false, t.getClass().getSimpleName() + ": " + safeMessage(t), null, null, null, null, null, null, null, null);
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
        String explicit = System.getProperty("synaptik.apple.mps.lib");
        if (explicit != null && !explicit.isBlank()) {
            return SymbolLookup.libraryLookup(explicit.trim(), arena);
        }
        String envLib = System.getenv("SYNAPTIK_APPLE_MPS_LIB");
        if (envLib != null && !envLib.isBlank()) {
            return SymbolLookup.libraryLookup(envLib.trim(), arena);
        }
        return SymbolLookup.libraryLookup("synaptik_apple_mps", arena);
    }

    private static int appleDataTypeCode(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> 1;
            case BOOL -> 2;
            default -> throw new IllegalArgumentException("Apple DAG bridge currently supports FLOAT32/BOOL external inputs, got " + dataType);
        };
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
        @SuppressWarnings("unused")
        private final MethodHandle availableFn;
        @SuppressWarnings("unused")
        private final MethodHandle unavailableReasonFn;
        private final MethodHandle createContextFn;
        private final MethodHandle compilePartitionFn;
        private final MethodHandle executePartitionFn;
        private final MethodHandle destroyContextFn;
        private final MethodHandle destroyExecutableFn;

        private State(
                boolean available,
                String reason,
                Arena arenaRef,
                MethodHandle availableFn,
                MethodHandle unavailableReasonFn,
                MethodHandle createContextFn,
                MethodHandle compilePartitionFn,
                MethodHandle executePartitionFn,
                MethodHandle destroyContextFn,
                MethodHandle destroyExecutableFn
        ) {
            this.available = available;
            this.reason = reason;
            this.arenaRef = arenaRef;
            this.availableFn = availableFn;
            this.unavailableReasonFn = unavailableReasonFn;
            this.createContextFn = createContextFn;
            this.compilePartitionFn = compilePartitionFn;
            this.executePartitionFn = executePartitionFn;
            this.destroyContextFn = destroyContextFn;
            this.destroyExecutableFn = destroyExecutableFn;
        }
    }
}
