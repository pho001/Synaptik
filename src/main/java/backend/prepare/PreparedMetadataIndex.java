package backend.prepare;

import graph.execution.plan.CompiledNodeExecutionMetadata;

import java.util.HashMap;
import java.util.Map;

final class PreparedMetadataIndex {
    private final Map<Integer, CompiledNodeExecutionMetadata> metadataByNodeId;

    PreparedMetadataIndex() {
        this.metadataByNodeId = new HashMap<>();
    }

    private PreparedMetadataIndex(Map<Integer, CompiledNodeExecutionMetadata> metadataByNodeId) {
        this.metadataByNodeId = new HashMap<>(metadataByNodeId);
    }

    CompiledNodeExecutionMetadata metadataFor(int nodeId) {
        return metadataByNodeId.get(nodeId);
    }

    void publish(int nodeId, CompiledNodeExecutionMetadata metadata) {
        if (metadata == null) {
            metadataByNodeId.remove(nodeId);
            return;
        }
        metadataByNodeId.put(nodeId, metadata);
    }

    PreparedMetadataIndex fork() {
        return new PreparedMetadataIndex(metadataByNodeId);
    }
}
