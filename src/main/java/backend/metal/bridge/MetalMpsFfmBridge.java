package backend.metal.bridge;

import tensor.TensorInternalAccess;

import backend.metal.MetalMpsCapabilities;
import runtime.device.buffer.AcceleratorLayoutAbiV2Descriptor;
import runtime.device.buffer.AcceleratorLayoutAbiV2Support;
import backend.metal.lowering.MetalPartitionPlan;
import backend.accelerator.lowering.AcceleratorSubgraphSignature;
import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.metal.buffer.MetalBufferAllocator;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.buffer.MetalBufferHandle;
import backend.metal.buffer.MetalBufferAccess;
import tensor.DataType;
import tensor.Tensor;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Metal MPSGraph bridge backed by the Java Foreign Function and Memory API.
 *
 * <p>Availability is discovered once at class initialization by looking up the
 * native shim symbols. Compilation caches native executables by lowered subgraph
 * signature; bridge failures are surfaced as unavailable records so callers can
 * fall back to CPU execution.</p>
 */
public final class MetalMpsFfmBridge implements MetalMpsGraphBridge {
    private static final State STATE = init();
    private static final ConcurrentMap<AcceleratorSubgraphSignature, MemorySegment> EXECUTABLE_CACHE = new ConcurrentHashMap<>();
    private static final ConcurrentMap<Long, OutputBufferWriteDecision> OUTPUT_BUFFER_WRITE_DECISION_CACHE = new ConcurrentHashMap<>();
    private static volatile MetalMpsBridgeContext SHARED_CONTEXT;

    /**
     * Returns whether the Metal shim symbols required for bridge discovery were found.
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
        return STATE.reason == null ? "" : STATE.reason;
    }

    /**
     * Returns layered Metal native bridge capability state.
     */
    @Override
    public MetalMpsBridgeCapabilities capabilities() {
        return STATE.capabilities;
    }

    /**
     * Creates or returns the shared Metal context; failures return an unavailable context.
     */
    @Override
    public MetalMpsBridgeContext createContext() {
        MetalMpsBridgeContext cached = SHARED_CONTEXT;
        if (cached != null) {
            return cached;
        }
        if (!STATE.available) {
            return MetalMpsBridgeContext.unavailable(unavailableReason());
        }
        if (STATE.createContextFn == null) {
            return MetalMpsBridgeContext.unavailable("Metal MPS shim is available, but create_context ABI is missing.");
        }
        try {
            MemorySegment handle = (MemorySegment) STATE.createContextFn.invokeExact();
            if (handle == null || handle.equals(MemorySegment.NULL)) {
                return MetalMpsBridgeContext.unavailable("Metal MPS shim returned a null context handle.");
            }
            MetalMpsBridgeContext created = new MetalMpsBridgeContext(true, handle, "");
            SHARED_CONTEXT = created;
            return created;
        } catch (Throwable t) {
            return MetalMpsBridgeContext.unavailable("create_context failed: " + safeMessage(t));
        }
    }

    /**
     * Compiles a lowered Metal partition through the MPS shim and executable cache.
     */
    @Override
    public MetalMpsBridgeExecutable compile(MetalMpsBridgeContext bridgeContext, MetalPartitionPlan plan) {
        if (!STATE.available) {
            return MetalMpsBridgeExecutable.unavailable(unavailableReason());
        }
        if (bridgeContext == null || !bridgeContext.available()) {
            return MetalMpsBridgeExecutable.unavailable(bridgeContext == null ? "Missing Metal bridge context." : bridgeContext.reason());
        }
        if (plan == null || plan.lowering() == null || plan.lowering().dagSpec() == null) {
            return MetalMpsBridgeExecutable.unavailable("Metal MPS FFM bridge requires lowered DAG spec.");
        }
        if (STATE.compilePartitionFn == null) {
            return MetalMpsBridgeExecutable.unavailable("Metal MPS shim is available, but compile_partition ABI is missing.");
        }
        try {
            AcceleratorDagSpec dagSpec = plan.lowering().dagSpec();
            boolean requiresDTypeV3Compile = requiresDTypeV3Compile(dagSpec);
            boolean requiresAttributeV4Compile = requiresAttributeV4Compile(dagSpec);
            if (requiresDTypeV3Compile && STATE.compilePartitionDTypeV3Fn == null) {
                return MetalMpsBridgeExecutable.unavailable("Metal MPS dtype ABI v3 compile symbol is unavailable for widened dtype DAG.");
            }
            if (requiresAttributeV4Compile && STATE.compilePartitionDTypeV4Fn == null) {
                return MetalMpsBridgeExecutable.unavailable("Metal MPS dtype/attribute ABI v4 compile symbol is unavailable for attributed layout/index DAG.");
            }
            AcceleratorSubgraphSignature signature = AcceleratorSubgraphSignature.from(plan);
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
                            externalInputDTypes[i] = requiresDTypeV3Compile
                                    ? MetalMpsCapabilities.abiDescriptorDataTypeCode(input.dataType())
                                    : MetalMpsCapabilities.abiDataTypeCode(input.dataType());
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
                        int[] input4Kinds = new int[nodeCount];
                        int[] input4Indices = new int[nodeCount];
                        float[] scalarValues = new float[nodeCount];
                        int[] nodeAttributes = new int[nodeCount * 8];
                        int[] outputRanks = new int[nodeCount];
                        int[] outputDim0 = new int[nodeCount];
                        int[] outputDim1 = new int[nodeCount];
                        int[] outputDim2 = new int[nodeCount];
                        int[] outputDim3 = new int[nodeCount];
                        int[] nodeOutputDTypes = new int[nodeCount];
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
                            input4Kinds[i] = node.input4().kind().abiCode();
                            input4Indices[i] = node.input4().index();
                            scalarValues[i] = Float.intBitsToFloat(node.scalarValueBits());
                            int attrBase = i * 8;
                            nodeAttributes[attrBase] = node.attribute0();
                            nodeAttributes[attrBase + 1] = node.attribute1();
                            nodeAttributes[attrBase + 2] = node.attribute2();
                            nodeAttributes[attrBase + 3] = node.attribute3();
                            nodeAttributes[attrBase + 4] = node.attribute4();
                            nodeAttributes[attrBase + 5] = node.attribute5();
                            nodeAttributes[attrBase + 6] = node.attribute6();
                            nodeAttributes[attrBase + 7] = node.attribute7();
                            outputRanks[i] = node.outputRank();
                            outputDim0[i] = node.outputDim0();
                            outputDim1[i] = node.outputDim1();
                            outputDim2[i] = node.outputDim2();
                            outputDim3[i] = node.outputDim3();
                            nodeOutputDTypes[i] = requiresDTypeV3Compile
                                    ? MetalMpsCapabilities.abiDescriptorDataTypeCode(node.outputDataType())
                                    : MetalMpsCapabilities.abiDataTypeCode(node.outputDataType());
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
                        MemorySegment input4KindsSeg = input4Kinds.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, input4Kinds);
                        MemorySegment input4IndicesSeg = input4Indices.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, input4Indices);
                        MemorySegment scalarValuesSeg = scalarValues.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_FLOAT, scalarValues);
                        MemorySegment nodeAttributesSeg = nodeAttributes.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, nodeAttributes);
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
                        MemorySegment nodeOutputDTypesSeg = nodeOutputDTypes.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, nodeOutputDTypes);
                        int outputCount = dagSpec.outputNodeIndices().size();
                        int[] outputNodeIndices = dagSpec.outputNodeIndices().stream().mapToInt(Integer::intValue).toArray();
                        MemorySegment outputNodeIndicesSeg = outputNodeIndices.length == 0
                                ? MemorySegment.NULL
                                : compileArena.allocateFrom(JAVA_INT, outputNodeIndices);
                        if (requiresAttributeV4Compile) {
                            return (MemorySegment) STATE.compilePartitionDTypeV4Fn.invokeExact(
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
                                input4KindsSeg,
                                input4IndicesSeg,
                                scalarValuesSeg,
                                nodeAttributesSeg,
                                outputRanksSeg,
                                outputDim0Seg,
                                outputDim1Seg,
                                outputDim2Seg,
                                outputDim3Seg,
                                nodeOutputDTypesSeg,
                                outputCount,
                                outputNodeIndicesSeg
                            );
                        }
                        if (requiresDTypeV3Compile) {
                            return (MemorySegment) STATE.compilePartitionDTypeV3Fn.invokeExact(
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
                                input4KindsSeg,
                                input4IndicesSeg,
                                scalarValuesSeg,
                                outputRanksSeg,
                                outputDim0Seg,
                                outputDim1Seg,
                                outputDim2Seg,
                                outputDim3Seg,
                                nodeOutputDTypesSeg,
                                outputCount,
                                outputNodeIndicesSeg
                            );
                        }
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
                            input4KindsSeg,
                            input4IndicesSeg,
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
                return MetalMpsBridgeExecutable.unavailable("Metal MPS shim returned a null executable handle.");
            }
            return new MetalMpsBridgeExecutable(
                    true,
                    handle,
                    "",
                    cacheHit,
                    dagSpec.externalInputs().stream().map(AcceleratorDagInput::nodeId).toList(),
                    dagSpec.externalInputs().stream().map(AcceleratorDagInput::dataType).toList(),
                    dagSpec.outputNodeIds(),
                    dagSpec.outputNodeIndices().stream()
                            .map(index -> dagSpec.nodes().get(index).outputDataType())
                            .toList(),
                    dagSpec.outputNodeIndices()
            );
        } catch (Throwable t) {
            return MetalMpsBridgeExecutable.unavailable("compile_partition failed: " + safeMessage(t));
        }
    }

    private static boolean requiresDTypeV3Compile(AcceleratorDagSpec dagSpec) {
        if (dagSpec == null) {
            return false;
        }
        for (AcceleratorDagInput input : dagSpec.externalInputs()) {
            if (input.dataType() != DataType.FLOAT32 && input.dataType() != DataType.BOOL) {
                return true;
            }
        }
        for (AcceleratorDagNode node : dagSpec.nodes()) {
            if (node.outputDataType() != DataType.FLOAT32) {
                return true;
            }
        }
        return false;
    }

    private static boolean requiresAttributeV4Compile(AcceleratorDagSpec dagSpec) {
        if (dagSpec == null) {
            return false;
        }
        for (AcceleratorDagNode node : dagSpec.nodes()) {
            if (node.hasAttributes()) {
                return true;
            }
            switch (node.type()) {
                case GATHER_AXIS, GATHER_AXIS_GRAD, SLICE, SLICE_BACKWARD, CONCAT, PAD, TILE,
                     UNFOLD_AXIS, UNFOLD2D, FOLD2D -> {
                    return true;
                }
                default -> {
                }
            }
        }
        return false;
    }

    /**
     * Releases a non-shared native context when the shim exposes a destroy function.
     */
    @Override
    public void destroyContext(MetalMpsBridgeContext bridgeContext) {
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
     * Keeps cached native executables alive for reuse within the shared bridge context.
     */
    @Override
    public void destroyExecutable(MetalMpsBridgeExecutable executable) {
        // Executables are cached by subgraph signature and reused within the shared bridge context.
    }

    /**
     * Returns whether all native buffer binding symbols are available.
     */
    @Override
    public boolean supportsBufferBindings() {
        return STATE.available
                && STATE.createBufferFn != null
                && STATE.writeBufferFn != null
                && STATE.readBufferFn != null
                && STATE.destroyBufferFn != null
                && STATE.executePartitionBuffersFn != null;
    }

    /**
     * Returns whether the native shim exposes the Phase 45 MPSGraph output-buffer write probe.
     */
    @Override
    public boolean supportsOutputBufferWriteProbe() {
        return supportsBufferBindings() && STATE.probeOutputBufferWriteBuffersFn != null;
    }

    @Override
    public boolean supportsLayoutMaterialization() {
        return STATE.available && STATE.layoutContiguousBufferFn != null;
    }

    /**
     * Creates a Metal buffer allocator backed by this FFM bridge.
     */
    @Override
    public MetalBufferAllocator createBufferAllocator(MetalMpsBridgeContext bridgeContext) {
        if (!supportsBufferBindings()) {
            return MetalBufferAllocator.unavailable("Metal MPS buffer ABI symbols are unavailable.");
        }
        if (bridgeContext == null || !bridgeContext.available()) {
            return MetalBufferAllocator.unavailable(bridgeContext == null ? "Missing Metal bridge context." : bridgeContext.reason());
        }
        return MetalBufferAllocator.available(new MetalBufferAllocator.NativeAccess() {
            @Override
            public MetalBufferHandle createBuffer(long byteLength, int storageMode, MemorySegment initialData, long initialDataBytes) {
                try {
                    MemorySegment initial = initialData == null ? MemorySegment.NULL : initialData;
                    MemorySegment handle = (MemorySegment) STATE.createBufferFn.invokeExact(
                            bridgeContext.handle(),
                            byteLength,
                            storageMode,
                            initial,
                            initialDataBytes
                    );
                    if (handle == null || handle.equals(MemorySegment.NULL)) {
                        throw new UnsupportedOperationException("Metal MPS create_buffer returned null.");
                    }
                    return new MetalBufferHandle(handle, byteLength, storageModeLabel(storageMode), "MetalMpsFfmBridge", true);
                } catch (Throwable t) {
                    throw new UnsupportedOperationException("Metal MPS create_buffer failed: " + safeMessage(t), t);
                }
            }

            @Override
            public void readBuffer(MetalBufferHandle handle, MemorySegment destination, long byteLength) {
                try {
                    int status = (int) STATE.readBufferFn.invokeExact(handle.nativeHandle(), destination, byteLength);
                    if (status != 0) {
                        throw new UnsupportedOperationException("Metal MPS read_buffer returned non-zero status: " + status);
                    }
                } catch (Throwable t) {
                    throw new UnsupportedOperationException("Metal MPS read_buffer failed: " + safeMessage(t), t);
                }
            }

            @Override
            public void destroyBuffer(MetalBufferHandle handle) {
                try {
                    STATE.destroyBufferFn.invokeExact(handle.nativeHandle());
                } catch (Throwable t) {
                    throw new UnsupportedOperationException("Metal MPS destroy_buffer failed: " + safeMessage(t), t);
                }
            }
        });
    }

    @Override
    public void materializeLayout(
            MetalMpsBridgeContext context,
            MetalBufferBinding source,
            MetalBufferBinding destination
    ) {
        if (!supportsLayoutMaterialization()) {
            throw new UnsupportedOperationException("Metal layout materialization symbol is unavailable.");
        }
        if (context == null || !context.available()) {
            throw new UnsupportedOperationException(context == null ? "Missing Metal bridge context." : context.reason());
        }
        if (source == null || !source.available() || destination == null || !destination.available()) {
            throw new UnsupportedOperationException("Metal layout materialization requires available source and destination bindings.");
        }
        AcceleratorLayoutAbiV2Descriptor descriptor = AcceleratorLayoutAbiV2Descriptor.fromBinding(source);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment shape = longArray(arena, descriptor.shape());
            MemorySegment strides = longArray(arena, descriptor.strides());
            int status = (int) STATE.layoutContiguousBufferFn.invokeExact(
                    context.handle(),
                    source.handle().nativeHandle(),
                    destination.handle().nativeHandle(),
                    MetalMpsCapabilities.abiDescriptorDataTypeCode(descriptor.dataType()),
                    descriptor.rank(),
                    shape,
                    strides,
                    (long) descriptor.storageOffset(),
                    descriptor.logicalElementCount(),
                    descriptor.physicalByteSpan(),
                    destination.logicalByteLength()
            );
            if (status != 0) {
                throw new UnsupportedOperationException("Metal layout materialization returned non-zero status: " + status);
            }
        } catch (Throwable t) {
            throw new UnsupportedOperationException("Metal layout materialization failed: " + safeMessage(t), t);
        }
    }

    /**
     * Executes a compiled Metal DAG and copies native output buffers back into runtime tensors.
     */
    @Override
    public MetalMpsBridgeExecutionStats execute(
            MetalMpsBridgeContext bridgeContext,
            MetalMpsBridgeExecutable executable,
            java.util.List<Tensor> externalInputs,
            java.util.List<Tensor> outputs
    ) {
        if (!STATE.available) {
            throw new UnsupportedOperationException(unavailableReason());
        }
        if (bridgeContext == null || !bridgeContext.available()) {
            throw new UnsupportedOperationException(bridgeContext == null ? "Missing Metal bridge context." : bridgeContext.reason());
        }
        if (executable == null || !executable.available()) {
            throw new UnsupportedOperationException(executable == null ? "Missing Metal bridge executable." : executable.reason());
        }
        if (STATE.executePartitionFn == null) {
            throw new UnsupportedOperationException("Metal MPS shim is available, but execute_partition ABI is missing.");
        }
        java.util.List<Tensor> resolvedExternalInputs = externalInputs == null ? java.util.List.of() : java.util.List.copyOf(externalInputs);
        java.util.List<Tensor> resolvedOutputs = outputs == null ? java.util.List.of() : java.util.List.copyOf(outputs);
        if (resolvedOutputs.isEmpty()) {
            throw new UnsupportedOperationException("Metal MPS FFM bridge requires at least one output tensor.");
        }
        long totalStart = System.nanoTime();
        long inputBytes = byteSize(resolvedExternalInputs);
        long outputBytes = byteSize(resolvedOutputs);
        long javaToNativeCopyNs = 0L;
        long outputAllocationNs = 0L;
        long nativeExecuteNs = 0L;
        long nativeToJavaCopyNs = 0L;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment externalInputSegs = resolvedExternalInputs.isEmpty() ? MemorySegment.NULL : arena.allocate(ADDRESS, resolvedExternalInputs.size());
            for (int i = 0; i < resolvedExternalInputs.size(); i++) {
                Tensor tensor = resolvedExternalInputs.get(i);
                if (tensor == null) {
                    throw new UnsupportedOperationException("Metal MPS FFM bridge received null external input.");
                }
                MemorySegment dataSeg;
                long copyStart = System.nanoTime();
                dataSeg = switch (tensor.getDataType()) {
                    case FLOAT32 -> arena.allocateFrom(JAVA_FLOAT, TensorInternalAccess.float32Data(tensor));
                    case BOOL -> arena.allocateFrom(JAVA_BYTE, TensorInternalAccess.boolData(tensor));
                    default -> throw new UnsupportedOperationException(MetalMpsCapabilities.unsupportedDTypeMessage(tensor.getDataType()));
                };
                javaToNativeCopyNs += System.nanoTime() - copyStart;
                externalInputSegs.setAtIndex(ADDRESS, i, dataSeg);
            }
            MemorySegment outputSegs = arena.allocate(ADDRESS, resolvedOutputs.size());
            MemorySegment outputLengths = arena.allocate(JAVA_INT, resolvedOutputs.size());
            for (int i = 0; i < resolvedOutputs.size(); i++) {
                Tensor out = resolvedOutputs.get(i);
                float[] outData = out == null ? null : TensorInternalAccess.float32Data(out);
                if (outData == null) {
                    throw new UnsupportedOperationException("Metal MPS FFM bridge currently supports only FLOAT32 output tensors with direct float[] storage.");
                }
                long allocationStart = System.nanoTime();
                MemorySegment outSeg = arena.allocate(JAVA_FLOAT, outData.length);
                outputAllocationNs += System.nanoTime() - allocationStart;
                outputSegs.setAtIndex(ADDRESS, i, outSeg);
                outputLengths.setAtIndex(JAVA_INT, i, outData.length);
            }
            long nativeStart = System.nanoTime();
            int status = (int) STATE.executePartitionFn.invokeExact(
                    bridgeContext.handle(),
                    executable.handle(),
                    externalInputSegs,
                    resolvedExternalInputs.size(),
                    outputSegs,
                    resolvedOutputs.size()
            );
            nativeExecuteNs = System.nanoTime() - nativeStart;
            if (status == 0) {
                for (int i = 0; i < resolvedOutputs.size(); i++) {
                    Tensor out = resolvedOutputs.get(i);
                    float[] outData = TensorInternalAccess.float32Data(out);
                    long copyStart = System.nanoTime();
                    MemorySegment.ofArray(outData).copyFrom(outputSegs.getAtIndex(ADDRESS, i).reinterpret((long) outData.length * JAVA_FLOAT.byteSize()));
                    nativeToJavaCopyNs += System.nanoTime() - copyStart;
                }
                return new MetalMpsBridgeExecutionStats(
                        false,
                        "",
                        MetalMpsBridgeExecutionPath.TENSOR_ARRAY_COPY,
                        MetalNativeCopyStrategy.MPSGRAPH_RESULT_COPY,
                        resolvedExternalInputs.size(),
                        resolvedOutputs.size(),
                        inputBytes,
                        outputBytes,
                        javaToNativeCopyNs,
                        outputAllocationNs,
                        nativeExecuteNs,
                        0L,
                        nativeToJavaCopyNs,
                        System.nanoTime() - totalStart
                );
            }
            throw new UnsupportedOperationException("Metal MPS execute_partition returned non-zero status: " + status);
        } catch (Throwable t) {
            throw new UnsupportedOperationException("Metal MPS execute_partition failed: " + safeMessage(t), t);
        }
    }

    /**
     * Executes a compiled Metal DAG using preallocated native buffer bindings.
     */
    @Override
    public MetalMpsBridgeExecutionStats executeBuffers(
            MetalMpsBridgeContext bridgeContext,
            MetalMpsBridgeExecutable executable,
            java.util.List<MetalBufferBinding> externalInputs,
            java.util.List<MetalBufferBinding> outputs
    ) {
        if (!supportsBufferBindings()) {
            throw new UnsupportedOperationException("Metal MPS buffer ABI symbols are unavailable.");
        }
        if (bridgeContext == null || !bridgeContext.available()) {
            throw new UnsupportedOperationException(bridgeContext == null ? "Missing Metal bridge context." : bridgeContext.reason());
        }
        if (executable == null || !executable.available()) {
            throw new UnsupportedOperationException(executable == null ? "Missing Metal bridge executable." : executable.reason());
        }
        java.util.List<MetalBufferBinding> resolvedExternalInputs = externalInputs == null ? java.util.List.of() : java.util.List.copyOf(externalInputs);
        java.util.List<MetalBufferBinding> resolvedOutputs = outputs == null ? java.util.List.of() : java.util.List.copyOf(outputs);
        validateBufferBindings(executable, resolvedExternalInputs, resolvedOutputs);

        long proofKey = executable.handle().address();
        OutputBufferWriteDecision decision = OUTPUT_BUFFER_WRITE_DECISION_CACHE.get(proofKey);
        if (decision != null && decision.useTrueOutputBufferWrite()) {
            return executeBuffersNoCopy(bridgeContext, executable, resolvedExternalInputs, resolvedOutputs);
        }
        if (decision == null && supportsOutputBufferWriteProbe()) {
            MetalOutputBufferWriteProbeResult proof = probeOutputBufferWriteContract(
                    bridgeContext,
                    executable,
                    resolvedExternalInputs,
                    resolvedOutputs
            );
            OutputBufferWriteDecision resolvedDecision = OutputBufferWriteDecision.from(proof);
            OutputBufferWriteDecision existingDecision = OUTPUT_BUFFER_WRITE_DECISION_CACHE.putIfAbsent(proofKey, resolvedDecision);
            if (existingDecision != null) {
                if (existingDecision.useTrueOutputBufferWrite()) {
                    return executeBuffersNoCopy(bridgeContext, executable, resolvedExternalInputs, resolvedOutputs);
                }
                return executeBuffersCopied(bridgeContext, executable, resolvedExternalInputs, resolvedOutputs);
            }
            if (resolvedDecision.useTrueOutputBufferWrite()) {
                return trueOutputBufferWriteStats(proof.probeExecutionStats());
            }
        }
        return executeBuffersCopied(bridgeContext, executable, resolvedExternalInputs, resolvedOutputs);
    }

    private MetalMpsBridgeExecutionStats executeBuffersCopied(
            MetalMpsBridgeContext bridgeContext,
            MetalMpsBridgeExecutable executable,
            java.util.List<MetalBufferBinding> resolvedExternalInputs,
            java.util.List<MetalBufferBinding> resolvedOutputs
    ) {
        long totalStart = System.nanoTime();
        long inputBytes = byteSizeBindings(resolvedExternalInputs);
        long outputBytes = byteSizeBindings(resolvedOutputs);
        long nativeExecuteNs;
        long nativeDeviceCopyNs;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment inputHandles = resolvedExternalInputs.isEmpty()
                    ? MemorySegment.NULL
                    : arena.allocate(ADDRESS, resolvedExternalInputs.size());
            for (int i = 0; i < resolvedExternalInputs.size(); i++) {
                inputHandles.setAtIndex(ADDRESS, i, resolvedExternalInputs.get(i).handle().nativeHandle());
            }
            MemorySegment outputHandles = arena.allocate(ADDRESS, resolvedOutputs.size());
            for (int i = 0; i < resolvedOutputs.size(); i++) {
                outputHandles.setAtIndex(ADDRESS, i, resolvedOutputs.get(i).handle().nativeHandle());
            }
            MemorySegment nativeDeviceCopyNsOut = arena.allocate(JAVA_LONG);
            nativeDeviceCopyNsOut.set(JAVA_LONG, 0L, 0L);
            long nativeStart = System.nanoTime();
            int status = (int) STATE.executePartitionBuffersFn.invokeExact(
                    bridgeContext.handle(),
                    executable.handle(),
                    inputHandles,
                    resolvedExternalInputs.size(),
                    outputHandles,
                    resolvedOutputs.size(),
                    nativeDeviceCopyNsOut
            );
            nativeExecuteNs = System.nanoTime() - nativeStart;
            nativeDeviceCopyNs = nativeDeviceCopyNsOut.get(JAVA_LONG, 0L);
            if (status != 0) {
                throw new UnsupportedOperationException("Metal MPS execute_partition_f32_buffers returned non-zero status: " + status);
            }
        } catch (Throwable t) {
            throw new UnsupportedOperationException("Metal MPS execute_partition_f32_buffers failed: " + safeMessage(t), t);
        }
        return new MetalMpsBridgeExecutionStats(
                false,
                "",
                MetalMpsBridgeExecutionPath.BUFFER_BINDING,
                MetalNativeCopyStrategy.MPSGRAPH_RESULT_COPY,
                resolvedExternalInputs.size(),
                resolvedOutputs.size(),
                inputBytes,
                outputBytes,
                0L,
                0L,
                nativeExecuteNs,
                nativeDeviceCopyNs,
                0L,
                System.nanoTime() - totalStart
        );
    }

    private MetalMpsBridgeExecutionStats executeBuffersNoCopy(
            MetalMpsBridgeContext bridgeContext,
            MetalMpsBridgeExecutable executable,
            java.util.List<MetalBufferBinding> resolvedExternalInputs,
            java.util.List<MetalBufferBinding> resolvedOutputs
    ) {
        MetalMpsBridgeExecutionStats stats = probeOutputBufferWriteWithoutResultCopy(
                bridgeContext,
                executable,
                resolvedExternalInputs,
                resolvedOutputs
        );
        return trueOutputBufferWriteStats(stats);
    }

    /**
     * Executes the MPSGraph buffer path without copying returned MPSNDArray storage into caller outputs.
     *
     * <p>This is a test/proof seam for Phase 45. Normal production execution must continue to use
     * {@link #executeBuffers(MetalMpsBridgeContext, MetalMpsBridgeExecutable, java.util.List, java.util.List)}
     * unless this probe proves a route-specific true output-buffer write contract.</p>
     */
    MetalMpsBridgeExecutionStats probeOutputBufferWriteWithoutResultCopy(
            MetalMpsBridgeContext bridgeContext,
            MetalMpsBridgeExecutable executable,
            java.util.List<MetalBufferBinding> externalInputs,
            java.util.List<MetalBufferBinding> outputs
    ) {
        if (!supportsOutputBufferWriteProbe()) {
            throw new UnsupportedOperationException("Metal MPS output-buffer write probe symbol is unavailable.");
        }
        if (bridgeContext == null || !bridgeContext.available()) {
            throw new UnsupportedOperationException(bridgeContext == null ? "Missing Metal bridge context." : bridgeContext.reason());
        }
        if (executable == null || !executable.available()) {
            throw new UnsupportedOperationException(executable == null ? "Missing Metal bridge executable." : executable.reason());
        }
        java.util.List<MetalBufferBinding> resolvedExternalInputs = externalInputs == null ? java.util.List.of() : java.util.List.copyOf(externalInputs);
        java.util.List<MetalBufferBinding> resolvedOutputs = outputs == null ? java.util.List.of() : java.util.List.copyOf(outputs);
        validateBufferBindings(executable, resolvedExternalInputs, resolvedOutputs);

        long totalStart = System.nanoTime();
        long inputBytes = byteSizeBindings(resolvedExternalInputs);
        long outputBytes = byteSizeBindings(resolvedOutputs);
        long nativeExecuteNs;
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment inputHandles = resolvedExternalInputs.isEmpty()
                    ? MemorySegment.NULL
                    : arena.allocate(ADDRESS, resolvedExternalInputs.size());
            for (int i = 0; i < resolvedExternalInputs.size(); i++) {
                inputHandles.setAtIndex(ADDRESS, i, resolvedExternalInputs.get(i).handle().nativeHandle());
            }
            MemorySegment outputHandles = arena.allocate(ADDRESS, resolvedOutputs.size());
            for (int i = 0; i < resolvedOutputs.size(); i++) {
                outputHandles.setAtIndex(ADDRESS, i, resolvedOutputs.get(i).handle().nativeHandle());
            }
            long nativeStart = System.nanoTime();
            int status = (int) STATE.probeOutputBufferWriteBuffersFn.invokeExact(
                    bridgeContext.handle(),
                    executable.handle(),
                    inputHandles,
                    resolvedExternalInputs.size(),
                    outputHandles,
                    resolvedOutputs.size()
            );
            nativeExecuteNs = System.nanoTime() - nativeStart;
            if (status != 0) {
                throw new UnsupportedOperationException("Metal MPS output-buffer write probe returned non-zero status: " + status);
            }
        } catch (Throwable t) {
            throw new UnsupportedOperationException("Metal MPS output-buffer write probe failed: " + safeMessage(t), t);
        }
        return new MetalMpsBridgeExecutionStats(
                false,
                "",
                MetalMpsBridgeExecutionPath.BUFFER_BINDING,
                MetalNativeCopyStrategy.UNKNOWN_OR_UNPROVEN,
                resolvedExternalInputs.size(),
                resolvedOutputs.size(),
                inputBytes,
                outputBytes,
                0L,
                0L,
                nativeExecuteNs,
                0L,
                0L,
                System.nanoTime() - totalStart
        );
    }

    /**
     * Runs a copied reference execution and a no-copy probe execution, then compares the output bytes.
     */
    @Override
    public MetalOutputBufferWriteProbeResult probeOutputBufferWriteContract(
            MetalMpsBridgeContext bridgeContext,
            MetalMpsBridgeExecutable executable,
            java.util.List<MetalBufferBinding> externalInputs,
            java.util.List<MetalBufferBinding> outputs
    ) {
        if (!supportsOutputBufferWriteProbe()) {
            return MetalOutputBufferWriteProbeResult.unsupported("Metal MPS output-buffer write probe symbol is unavailable.");
        }
        if (bridgeContext == null || !bridgeContext.available()) {
            return MetalOutputBufferWriteProbeResult.unsupported(
                    bridgeContext == null ? "Missing Metal bridge context." : bridgeContext.reason()
            );
        }
        if (executable == null || !executable.available()) {
            return MetalOutputBufferWriteProbeResult.unsupported(
                    executable == null ? "Missing Metal bridge executable." : executable.reason()
            );
        }
        java.util.List<MetalBufferBinding> resolvedExternalInputs = externalInputs == null
                ? java.util.List.of()
                : java.util.List.copyOf(externalInputs);
        java.util.List<MetalBufferBinding> resolvedOutputs = outputs == null
                ? java.util.List.of()
                : java.util.List.copyOf(outputs);
        try {
            validateBufferBindings(executable, resolvedExternalInputs, resolvedOutputs);
        } catch (UnsupportedOperationException ex) {
            return MetalOutputBufferWriteProbeResult.unsupported(ex.getMessage());
        }

        java.util.List<byte[]> sentinelBytes = readBindingBytes(resolvedOutputs);
        java.util.List<MetalBufferBinding> copiedOutputs = new ArrayList<>(resolvedOutputs.size());
        try {
            for (MetalBufferBinding output : resolvedOutputs) {
                copiedOutputs.add(scratchOutputBinding(bridgeContext, output));
            }
            MetalMpsBridgeExecutionStats copiedStats = executeBuffersCopied(
                    bridgeContext,
                    executable,
                    resolvedExternalInputs,
                    copiedOutputs
            );
            java.util.List<byte[]> copiedBytes = readBindingBytes(copiedOutputs);
            MetalMpsBridgeExecutionStats probeStats = probeOutputBufferWriteWithoutResultCopy(
                    bridgeContext,
                    executable,
                    resolvedExternalInputs,
                    resolvedOutputs
            );
            java.util.List<byte[]> probeBytes = readBindingBytes(resolvedOutputs);
            if (bytesEqual(copiedBytes, probeBytes)) {
                return new MetalOutputBufferWriteProbeResult(
                        MetalOutputBufferWriteProbeStatus.MATCHES_COPIED_RESULT,
                        "no-copy output bytes matched copied MPSGraph result bytes",
                        copiedStats,
                        probeStats
                );
            }
            if (bytesEqual(sentinelBytes, probeBytes)) {
                return new MetalOutputBufferWriteProbeResult(
                        MetalOutputBufferWriteProbeStatus.UNCHANGED_SENTINEL,
                        "no-copy execution left caller output buffers unchanged",
                        copiedStats,
                        probeStats
                );
            }
            return new MetalOutputBufferWriteProbeResult(
                    MetalOutputBufferWriteProbeStatus.MISMATCHED_RESULT,
                    "no-copy execution changed caller output buffers but did not match copied result bytes",
                    copiedStats,
                    probeStats
            );
        } catch (RuntimeException ex) {
            return MetalOutputBufferWriteProbeResult.unsupported("output-buffer write proof failed: " + safeMessage(ex));
        } finally {
            for (MetalBufferBinding scratch : copiedOutputs) {
                destroyScratchBuffer(scratch);
            }
        }
    }

    static void validateBufferBindings(
            MetalMpsBridgeExecutable executable,
            java.util.List<MetalBufferBinding> externalInputs,
            java.util.List<MetalBufferBinding> outputs
    ) {
        if (externalInputs.size() != executable.externalInputNodeIds().size()) {
            throw new UnsupportedOperationException("Metal buffer execution received "
                    + externalInputs.size() + " inputs, expected " + executable.externalInputNodeIds().size() + ".");
        }
        if (outputs.isEmpty() || outputs.size() != executable.outputNodeIds().size()) {
            throw new UnsupportedOperationException("Metal buffer execution received "
                    + outputs.size() + " outputs, expected " + executable.outputNodeIds().size() + ".");
        }
        for (int i = 0; i < externalInputs.size(); i++) {
            MetalBufferBinding binding = externalInputs.get(i);
            validateBinding(binding, "external input " + i);
            int expectedNodeId = executable.externalInputNodeIds().get(i);
            if (binding.nodeId() != expectedNodeId) {
                throw new UnsupportedOperationException("Metal buffer input " + i
                        + " nodeId " + binding.nodeId()
                        + " does not match executable nodeId " + expectedNodeId + ".");
            }
            DataType expected = i < executable.externalInputDataTypes().size() ? executable.externalInputDataTypes().get(i) : null;
            if (expected != null && binding.layout().dataType() != expected) {
                throw new UnsupportedOperationException("Metal buffer input " + i
                        + " dtype " + binding.layout().dataType() + " does not match executable dtype " + expected + ".");
            }
            if (binding.layout().dataType() != DataType.FLOAT32
                    && binding.layout().dataType() != DataType.BFLOAT16
                    && binding.layout().dataType() != DataType.BOOL
                    && binding.layout().dataType() != DataType.INT32) {
                throw new UnsupportedOperationException("Metal buffer input " + i
                        + " has unsupported dtype " + binding.layout().dataType() + ".");
            }
            if (!readable(binding.access())) {
                throw new UnsupportedOperationException("Metal buffer input " + i
                        + " access " + binding.access() + " is not readable.");
            }
        }
        for (int i = 0; i < outputs.size(); i++) {
            MetalBufferBinding binding = outputs.get(i);
            validateBinding(binding, "output " + i);
            int expectedNodeId = executable.outputNodeIds().get(i);
            if (binding.nodeId() != expectedNodeId) {
                throw new UnsupportedOperationException("Metal buffer output " + i
                        + " nodeId " + binding.nodeId()
                        + " does not match executable nodeId " + expectedNodeId + ".");
            }
            DataType expected = i < executable.outputDataTypes().size() ? executable.outputDataTypes().get(i) : DataType.FLOAT32;
            if (expected != null && binding.layout().dataType() != expected) {
                throw new UnsupportedOperationException("Metal buffer output " + i
                        + " dtype " + binding.layout().dataType() + " does not match executable dtype " + expected + ".");
            }
            if (binding.layout().dataType() != DataType.FLOAT32
                    && binding.layout().dataType() != DataType.BFLOAT16
                    && binding.layout().dataType() != DataType.BOOL
                    && binding.layout().dataType() != DataType.INT32
                    && binding.layout().dataType() != DataType.INT64) {
                throw new UnsupportedOperationException("Metal buffer outputs support FLOAT32/BFLOAT16/BOOL/INT32/INT64 only; got " + binding.layout().dataType() + ".");
            }
            if (!writable(binding.access())) {
                throw new UnsupportedOperationException("Metal buffer output " + i
                        + " access " + binding.access() + " is not writable.");
            }
        }
    }

    private static MetalMpsBridgeExecutionStats trueOutputBufferWriteStats(MetalMpsBridgeExecutionStats stats) {
        if (stats == null) {
            return null;
        }
        return new MetalMpsBridgeExecutionStats(
                stats.usedCpuFallback(),
                stats.fallbackReason(),
                stats.executionPath(),
                MetalNativeCopyStrategy.TRUE_OUTPUT_BUFFER_WRITE,
                stats.externalInputCount(),
                stats.outputCount(),
                stats.inputBytes(),
                stats.outputBytes(),
                stats.javaToNativeCopyNs(),
                stats.outputAllocationNs(),
                stats.nativeExecuteNs(),
                0L,
                stats.nativeToJavaCopyNs(),
                stats.totalNs()
        );
    }

    private static MetalBufferBinding scratchOutputBinding(
            MetalMpsBridgeContext bridgeContext,
            MetalBufferBinding output
    ) {
        try {
            MemorySegment handle = (MemorySegment) STATE.createBufferFn.invokeExact(
                    bridgeContext.handle(),
                    output.logicalByteLength(),
                    1,
                    MemorySegment.NULL,
                    0L
            );
            if (handle == null || handle.equals(MemorySegment.NULL)) {
                throw new UnsupportedOperationException("Metal MPS create_buffer returned null for scratch output.");
            }
            return new MetalBufferBinding(
                    output.nodeId(),
                    output.layout(),
                    new MetalBufferHandle(
                            handle,
                            output.logicalByteLength(),
                            storageModeLabel(1),
                            "MetalMpsFfmBridge:output-write-proof",
                            true
                    ),
                    MetalBufferAccess.READ_WRITE
            );
        } catch (Throwable t) {
            throw new UnsupportedOperationException("Metal MPS scratch output allocation failed: " + safeMessage(t), t);
        }
    }

    private static java.util.List<byte[]> readBindingBytes(java.util.List<MetalBufferBinding> bindings) {
        java.util.List<byte[]> out = new ArrayList<>(bindings.size());
        try (Arena arena = Arena.ofConfined()) {
            for (MetalBufferBinding binding : bindings) {
                int byteLength = Math.toIntExact(binding.logicalByteLength());
                MemorySegment destination = arena.allocate(byteLength);
                int status = (int) STATE.readBufferFn.invokeExact(
                        binding.handle().nativeHandle(),
                        destination,
                        (long) byteLength
                );
                if (status != 0) {
                    throw new UnsupportedOperationException("Metal MPS read_buffer returned non-zero status: " + status);
                }
                out.add(destination.toArray(JAVA_BYTE));
            }
        } catch (Throwable t) {
            throw new UnsupportedOperationException("Metal MPS read_buffer failed during output-buffer proof: " + safeMessage(t), t);
        }
        return java.util.List.copyOf(out);
    }

    private static boolean bytesEqual(java.util.List<byte[]> left, java.util.List<byte[]> right) {
        if (left.size() != right.size()) {
            return false;
        }
        for (int i = 0; i < left.size(); i++) {
            if (!Arrays.equals(left.get(i), right.get(i))) {
                return false;
            }
        }
        return true;
    }

    private static void destroyScratchBuffer(MetalBufferBinding binding) {
        if (binding == null || binding.handle() == null || !binding.handle().ownsHandle()) {
            return;
        }
        try {
            STATE.destroyBufferFn.invokeExact(binding.handle().nativeHandle());
        } catch (Throwable ignored) {
        }
    }

    private record OutputBufferWriteDecision(
            MetalOutputBufferWriteProbeStatus status,
            boolean useTrueOutputBufferWrite
    ) {
        private static OutputBufferWriteDecision from(MetalOutputBufferWriteProbeResult proof) {
            if (proof == null) {
                return new OutputBufferWriteDecision(MetalOutputBufferWriteProbeStatus.UNSUPPORTED, false);
            }
            return new OutputBufferWriteDecision(
                    proof.status(),
                    proof.provenTrueOutputBufferWrite() && probeIsNotSlowerThanCopiedPath(proof)
            );
        }

        private static boolean probeIsNotSlowerThanCopiedPath(MetalOutputBufferWriteProbeResult proof) {
            MetalMpsBridgeExecutionStats copied = proof.copiedExecutionStats();
            MetalMpsBridgeExecutionStats probe = proof.probeExecutionStats();
            if (copied == null || probe == null) {
                return false;
            }
            long copiedNs = copied.totalNs();
            long probeNs = probe.totalNs();
            if (copiedNs <= 0L || probeNs <= 0L) {
                return true;
            }
            // MPSGraph can legally write caller outputs but still run slower than its internal result path.
            long toleranceNs = Math.max(250_000L, copiedNs / 20L);
            return probeNs <= copiedNs + toleranceNs;
        }
    }

    private static void validateBinding(MetalBufferBinding binding, String role) {
        if (binding == null) {
            throw new UnsupportedOperationException("Metal buffer " + role + " binding is null.");
        }
        if (!binding.available()) {
            throw new UnsupportedOperationException("Metal buffer " + role + " binding is unavailable: " + binding.describe());
        }
    }

    private static boolean readable(MetalBufferAccess access) {
        return access == MetalBufferAccess.READ || access == MetalBufferAccess.READ_WRITE;
    }

    private static boolean writable(MetalBufferAccess access) {
        return access == MetalBufferAccess.WRITE || access == MetalBufferAccess.READ_WRITE;
    }

    private static long byteSizeBindings(java.util.List<MetalBufferBinding> bindings) {
        long bytes = 0L;
        if (bindings == null) {
            return 0L;
        }
        for (MetalBufferBinding binding : bindings) {
            if (binding != null) {
                bytes += binding.logicalByteLength();
            }
        }
        return bytes;
    }

    private static String storageModeLabel(int storageMode) {
        return switch (storageMode) {
            case 1 -> "shared";
            case 2 -> "private";
            case 3 -> "managed";
            default -> "unknown(" + storageMode + ")";
        };
    }

    private static long byteSize(java.util.List<Tensor> tensors) {
        long bytes = 0L;
        if (tensors == null) {
            return 0L;
        }
        for (Tensor tensor : tensors) {
            if (tensor == null) {
                continue;
            }
            bytes += (long) tensor.getFlatDataSize() * elementByteSize(tensor.getDataType());
        }
        return bytes;
    }

    private static int elementByteSize(tensor.DataType dataType) {
        if (dataType == null) {
            return 0;
        }
        return switch (dataType) {
            case FLOAT64 -> Double.BYTES;
            case FLOAT32 -> Float.BYTES;
            case BFLOAT16 -> Short.BYTES;
            case BOOL -> Byte.BYTES;
            case INT32 -> Integer.BYTES;
            case INT64 -> Long.BYTES;
        };
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
                            ADDRESS,
                            ADDRESS,
                            JAVA_INT,
                            ADDRESS
                    )
            );
            MethodHandle compilePartitionDTypeV3Fn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_compile_partition_dtype_v3",
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
                            ADDRESS,
                            ADDRESS,
                            ADDRESS,
                            JAVA_INT,
                            ADDRESS
                    )
            );
            MethodHandle compilePartitionDTypeV4Fn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_compile_partition_dtype_v4",
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
            MethodHandle createBufferFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_create_buffer",
                    FunctionDescriptor.of(ADDRESS, ADDRESS, JAVA_LONG, JAVA_INT, ADDRESS, JAVA_LONG)
            );
            MethodHandle writeBufferFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_write_buffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG)
            );
            MethodHandle readBufferFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_read_buffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG)
            );
            MethodHandle destroyBufferFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_destroy_buffer",
                    FunctionDescriptor.ofVoid(ADDRESS)
            );
            MethodHandle executePartitionBuffersFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_execute_partition_f32_buffers",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, ADDRESS, JAVA_INT, ADDRESS)
            );
            MethodHandle probeOutputBufferWriteBuffersFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_probe_output_buffer_write_f32_buffers",
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
            MethodHandle layoutAbiVersionFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_layout_abi_version",
                    FunctionDescriptor.of(JAVA_INT)
            );
            MethodHandle validateLayoutAbiV2Fn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_validate_layout_abi_v2",
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
            MethodHandle dtypeAbiVersionFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_dtype_abi_version",
                    FunctionDescriptor.of(JAVA_INT)
            );
            MethodHandle validateDTypeAbiV3Fn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_validate_dtype_abi_v3",
                    FunctionDescriptor.of(JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, ADDRESS)
            );
            MethodHandle layoutContiguousBufferFn = optionalHandle(
                    linker,
                    lookup,
                    "synaptik_apple_mps_layout_contiguous_buffer",
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, ADDRESS, ADDRESS, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG)
            );
            int layoutAbiV2Version = abiVersion(layoutAbiVersionFn);
            boolean layoutAbiV2Supported = layoutAbiV2Version == AcceleratorLayoutAbiV2Support.REQUIRED_VERSION
                    && validateLayoutAbiV2Fn != null;
            int dtypeAbiV3Version = abiVersion(dtypeAbiVersionFn);
            boolean dtypeAbiV3Supported = dtypeAbiV3Version == MetalDTypeAbiV3Support.REQUIRED_VERSION
                    && validateDTypeAbiV3Fn != null;

            int available = (int) availableFn.invokeExact();
            boolean bufferSupported = createBufferFn != null
                    && readBufferFn != null
                    && destroyBufferFn != null
                    && executePartitionBuffersFn != null;
            if (available != 0) {
                MetalMpsBridgeCapabilities capabilities = new MetalMpsBridgeCapabilities(
                        true,
                        true,
                        createContextFn != null,
                        compilePartitionFn != null && executePartitionFn != null,
                        bufferSupported,
                        layoutAbiV2Supported,
                        layoutAbiV2Version,
                        dtypeAbiV3Supported,
                        dtypeAbiV3Version,
                        metalCapabilityCode(layoutAbiV2Version, layoutAbiV2Supported, dtypeAbiV3Version, dtypeAbiV3Supported),
                        metalCapabilityReason(layoutAbiV2Version, layoutAbiV2Supported, dtypeAbiV3Version, dtypeAbiV3Supported)
                );
                return new State(true, null, arena, availableFn, unavailableReasonFn, createContextFn, compilePartitionFn,
                        compilePartitionDTypeV3Fn, compilePartitionDTypeV4Fn, executePartitionFn, createBufferFn, writeBufferFn, readBufferFn, destroyBufferFn,
                        executePartitionBuffersFn, probeOutputBufferWriteBuffersFn, destroyContextFn, destroyExecutableFn,
                        layoutContiguousBufferFn, capabilities);
            }

            MemorySegment reasonPtr = (MemorySegment) unavailableReasonFn.invokeExact();
            String reason = cStringOrDefault(reasonPtr, "Metal MPS shim reported unavailable.");
            MetalMpsBridgeCapabilities capabilities = new MetalMpsBridgeCapabilities(
                    true,
                    false,
                    false,
                    false,
                    bufferSupported,
                    layoutAbiV2Supported,
                    layoutAbiV2Version,
                    dtypeAbiV3Supported,
                    dtypeAbiV3Version,
                    MetalMpsCapabilityCode.RUNTIME_UNAVAILABLE,
                    reason
            );
            return new State(false, reason, arena, availableFn, unavailableReasonFn, createContextFn, compilePartitionFn,
                    compilePartitionDTypeV3Fn, compilePartitionDTypeV4Fn, executePartitionFn, createBufferFn, writeBufferFn, readBufferFn, destroyBufferFn,
                    executePartitionBuffersFn, probeOutputBufferWriteBuffersFn, destroyContextFn, destroyExecutableFn,
                    layoutContiguousBufferFn, capabilities);
        } catch (Throwable t) {
            String reason = t.getClass().getSimpleName() + ": " + safeMessage(t);
            return new State(false, t.getClass().getSimpleName() + ": " + safeMessage(t), null, null, null,
                    null, null, null, null, null, null, null, null, null, null, null, null, null, null,
                    MetalMpsBridgeCapabilities.unavailable(MetalMpsCapabilityCode.NATIVE_LIBRARY_UNAVAILABLE, reason));
        }
    }

    private static MemorySegment longArray(Arena arena, int[] values) {
        long[] out = new long[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i];
        }
        return out.length == 0 ? MemorySegment.NULL : arena.allocateFrom(JAVA_LONG, out);
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

    private static int abiVersion(MethodHandle abiVersionFn) {
        if (abiVersionFn == null) {
            return 0;
        }
        try {
            return Math.max(0, (int) abiVersionFn.invokeExact());
        } catch (Throwable ignored) {
            return 0;
        }
    }

    private static MetalMpsCapabilityCode metalCapabilityCode(
            int layoutAbiV2Version,
            boolean layoutAbiV2Supported,
            int dtypeAbiV3Version,
            boolean dtypeAbiV3Supported
    ) {
        if (layoutAbiV2Supported && dtypeAbiV3Supported) {
            return MetalMpsCapabilityCode.AVAILABLE;
        }
        if (!layoutAbiV2Supported) {
            if (layoutAbiV2Version == 0) {
                return MetalMpsCapabilityCode.LAYOUT_ABI_V2_UNAVAILABLE;
            }
            return MetalMpsCapabilityCode.LAYOUT_ABI_V2_VERSION_MISMATCH;
        }
        if (dtypeAbiV3Version == 0) {
            return MetalMpsCapabilityCode.DTYPE_ABI_V3_UNAVAILABLE;
        }
        return MetalMpsCapabilityCode.DTYPE_ABI_V3_VERSION_MISMATCH;
    }

    private static String metalCapabilityReason(
            int layoutAbiV2Version,
            boolean layoutAbiV2Supported,
            int dtypeAbiV3Version,
            boolean dtypeAbiV3Supported
    ) {
        if (layoutAbiV2Supported && dtypeAbiV3Supported) {
            return "";
        }
        if (!layoutAbiV2Supported) {
            if (layoutAbiV2Version == 0) {
                return "Metal layout ABI v2 symbols unavailable";
            }
            return "Metal layout ABI v2 version mismatch: expected "
                    + AcceleratorLayoutAbiV2Support.REQUIRED_VERSION
                    + ", got "
                    + layoutAbiV2Version;
        }
        if (dtypeAbiV3Version == 0) {
            return "Metal dtype ABI v3 symbols unavailable";
        }
        return "Metal dtype ABI v3 version mismatch: expected "
                + MetalDTypeAbiV3Support.REQUIRED_VERSION
                + ", got "
                + dtypeAbiV3Version;
    }

    private static SymbolLookup resolveLookup(Arena arena) {
        return MetalNativeLibraryResolver.resolveLookup(arena);
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
        private final MethodHandle compilePartitionDTypeV3Fn;
        private final MethodHandle compilePartitionDTypeV4Fn;
        private final MethodHandle executePartitionFn;
        private final MethodHandle createBufferFn;
        @SuppressWarnings("unused")
        private final MethodHandle writeBufferFn;
        private final MethodHandle readBufferFn;
        private final MethodHandle destroyBufferFn;
        private final MethodHandle executePartitionBuffersFn;
        private final MethodHandle probeOutputBufferWriteBuffersFn;
        private final MethodHandle destroyContextFn;
        private final MethodHandle destroyExecutableFn;
        private final MethodHandle layoutContiguousBufferFn;
        private final MetalMpsBridgeCapabilities capabilities;

        private State(
                boolean available,
                String reason,
                Arena arenaRef,
                MethodHandle availableFn,
                MethodHandle unavailableReasonFn,
                MethodHandle createContextFn,
                MethodHandle compilePartitionFn,
                MethodHandle compilePartitionDTypeV3Fn,
                MethodHandle compilePartitionDTypeV4Fn,
                MethodHandle executePartitionFn,
                MethodHandle createBufferFn,
                MethodHandle writeBufferFn,
                MethodHandle readBufferFn,
                MethodHandle destroyBufferFn,
                MethodHandle executePartitionBuffersFn,
                MethodHandle probeOutputBufferWriteBuffersFn,
                MethodHandle destroyContextFn,
                MethodHandle destroyExecutableFn,
                MethodHandle layoutContiguousBufferFn,
                MetalMpsBridgeCapabilities capabilities
        ) {
            this.available = available;
            this.reason = reason;
            this.arenaRef = arenaRef;
            this.availableFn = availableFn;
            this.unavailableReasonFn = unavailableReasonFn;
            this.createContextFn = createContextFn;
            this.compilePartitionFn = compilePartitionFn;
            this.compilePartitionDTypeV3Fn = compilePartitionDTypeV3Fn;
            this.compilePartitionDTypeV4Fn = compilePartitionDTypeV4Fn;
            this.executePartitionFn = executePartitionFn;
            this.createBufferFn = createBufferFn;
            this.writeBufferFn = writeBufferFn;
            this.readBufferFn = readBufferFn;
            this.destroyBufferFn = destroyBufferFn;
            this.executePartitionBuffersFn = executePartitionBuffersFn;
            this.probeOutputBufferWriteBuffersFn = probeOutputBufferWriteBuffersFn;
            this.destroyContextFn = destroyContextFn;
            this.destroyExecutableFn = destroyExecutableFn;
            this.layoutContiguousBufferFn = layoutContiguousBufferFn;
            this.capabilities = capabilities == null
                    ? MetalMpsBridgeCapabilities.unavailable(MetalMpsCapabilityCode.NATIVE_LIBRARY_UNAVAILABLE, this.reason)
                    : capabilities;
        }
    }
}
