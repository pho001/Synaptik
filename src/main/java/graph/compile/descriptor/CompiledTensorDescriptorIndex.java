package graph.compile.descriptor;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Immutable descriptor lookup for one compiled graph snapshot.
 */
public final class CompiledTensorDescriptorIndex {
    private static final CompiledTensorDescriptorIndex EMPTY = new CompiledTensorDescriptorIndex(List.of());

    private final List<CompiledTensorDescriptor> descriptors;
    private final Map<Integer, CompiledTensorDescriptor> byNodeId;

    public CompiledTensorDescriptorIndex(List<CompiledTensorDescriptor> descriptors) {
        this.descriptors = List.copyOf(descriptors == null ? List.of() : descriptors);
        this.byNodeId = this.descriptors.stream()
                .collect(Collectors.toUnmodifiableMap(
                        CompiledTensorDescriptor::nodeId,
                        descriptor -> descriptor
                ));
    }

    public static CompiledTensorDescriptorIndex empty() {
        return EMPTY;
    }

    public CompiledTensorDescriptor byNodeId(int nodeId) {
        CompiledTensorDescriptor descriptor = byNodeId.get(nodeId);
        if (descriptor == null) {
            throw new IllegalArgumentException("Unknown compiled tensor descriptor nodeId=" + nodeId);
        }
        return descriptor;
    }

    public CompiledTensorDescriptor input(int nodeId, int inputIndex) {
        CompiledTensorDescriptor descriptor = byNodeId(nodeId);
        if (inputIndex < 0 || inputIndex >= descriptor.inputIds().size()) {
            throw new IllegalArgumentException("Input index " + inputIndex + " is outside nodeId=" + nodeId);
        }
        return byNodeId(descriptor.inputIds().get(inputIndex));
    }

    public List<CompiledTensorDescriptor> inputs(int nodeId) {
        CompiledTensorDescriptor descriptor = byNodeId(nodeId);
        return descriptor.inputIds().stream()
                .map(this::byNodeId)
                .toList();
    }

    public List<CompiledTensorDescriptor> all() {
        return descriptors;
    }

    public boolean isEmpty() {
        return descriptors.isEmpty();
    }
}
