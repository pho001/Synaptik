package backend.cpu.fused.exec;

import backend.cpu.execution.CpuKernelContext;
import backend.memory.CpuMaterializationReason;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Per-dispatch native segment bindings for one fused CPU output.
 */
public record FusedNativeSegmentBindings(
        List<NativeTensorStorage> inputs,
        NativeTensorStorage output
) {
    private static final String OUTPUT_REASON = "CPU fused MemorySegment wrote output";

    public FusedNativeSegmentBindings {
        inputs = List.copyOf(inputs == null ? List.of() : inputs);
        output = Objects.requireNonNull(output, "output cannot be null");
    }

    public static FusedNativeSegmentBindings bind(CpuKernelContext context, List<Tensor> inputs, Tensor output) {
        Objects.requireNonNull(context, "context cannot be null");
        Objects.requireNonNull(inputs, "inputs cannot be null");
        Objects.requireNonNull(output, "output cannot be null");
        if (inputs.size() != context.inputNodeIds().size()) {
            throw new IllegalStateException("Fused native input count mismatch. tensors=" + inputs.size()
                    + ", nodeIds=" + context.inputNodeIds().size());
        }
        ArrayList<NativeTensorStorage> inputStorages = new ArrayList<>(inputs.size());
        for (int i = 0; i < inputs.size(); i++) {
            NativeTensorStorage storage = context.executionContext().requireNativeReadable(
                    context.inputNodeIds().get(i),
                    CpuMaterializationReason.CPU_CONSUMER
            );
            Tensor input = inputs.get(i);
            if (storage.getType() != input.getDataType()) {
                throw new IllegalStateException("Fused native input dtype mismatch at index=" + i
                        + ". tensorType=" + input.getDataType() + ", storageType=" + storage.getType());
            }
            inputStorages.add(storage);
        }
        NativeTensorStorage outputStorage = reusableOutputStorage(context, output);
        if (outputStorage == null) {
            outputStorage = context.executionContext().allocateNativeStorage(
                    output.getDataType(),
                    output.getFlatDataSize(),
                    "fused-node-" + context.nodeId() + ":" + output.getLabel()
            );
        }
        context.executionContext().reserveNativeOutputStorage(context.nodeId(), outputStorage);
        FusedNativeSegmentBindings bindings = new FusedNativeSegmentBindings(inputStorages, outputStorage);
        context.putRuntimeState(output, bindings);
        return bindings;
    }

    public static MemorySegment inputSegment(CpuKernelContext context, int inputIndex) {
        return require(context).inputSegment(inputIndex);
    }

    public static MemorySegment outputSegment(CpuKernelContext context) {
        return require(context).outputSegment();
    }

    public static NativeTensorStorage outputStorage(CpuKernelContext context) {
        return require(context).output();
    }

    public static void publish(CpuKernelContext context) {
        Objects.requireNonNull(context, "context cannot be null");
        NativeTensorStorage storage = outputStorage(context);
        storage.markModified();
        context.executionContext().attachNativeStorage(context.nodeId(), storage, OUTPUT_REASON);
    }

    public static void clear(CpuKernelContext context, Tensor output) {
        Objects.requireNonNull(context, "context cannot be null");
        if (output != null) {
            context.clearRuntimeState(output);
        }
    }

    public NativeTensorStorage inputStorage(int index) {
        if (index < 0 || index >= inputs.size()) {
            throw new IndexOutOfBoundsException("Fused native input index out of range: " + index);
        }
        NativeTensorStorage storage = inputs.get(index);
        storage.ensureOpen();
        return storage;
    }

    public MemorySegment inputSegment(int index) {
        return inputStorage(index).segment();
    }

    public MemorySegment outputSegment() {
        output.ensureOpen();
        return output.segment();
    }

    private static NativeTensorStorage reusableOutputStorage(CpuKernelContext context, Tensor output) {
        NativeTensorStorage storage = context.executionContext().nativeStorageForNodeId(context.nodeId());
        if (storage == null || storage.closed()) {
            return null;
        }
        if (storage.getType() != output.getDataType() || storage.getSize() != output.getFlatDataSize()) {
            return null;
        }
        if (!output.isContiguous() || output.getStorageOffsetUnsafe() != 0) {
            return null;
        }
        storage.ensureOpen();
        return storage;
    }

    private static FusedNativeSegmentBindings require(CpuKernelContext context) {
        Objects.requireNonNull(context, "context cannot be null");
        FusedNativeSegmentBindings bindings = context.runtimeStateFor(
                context.executionContext().runtimeTensorForNodeId(context.nodeId()),
                FusedNativeSegmentBindings.class
        );
        if (bindings == null) {
            throw new IllegalStateException("Missing fused native segment bindings for nodeId=" + context.nodeId());
        }
        return bindings;
    }
}
