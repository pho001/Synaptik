package prepare.context;

import runtime.execution.PreparedStepMetadata;

import java.util.HashMap;
import java.util.Map;

final class PreparedMetadataIndex {
    private final Map<Integer, PreparedStepMetadata> metadataByNodeId;

    PreparedMetadataIndex() {
        this.metadataByNodeId = new HashMap<>();
    }

    private PreparedMetadataIndex(Map<Integer, PreparedStepMetadata> metadataByNodeId) {
        this.metadataByNodeId = new HashMap<>(metadataByNodeId);
    }

    PreparedStepMetadata metadataFor(int nodeId) {
        return metadataByNodeId.get(nodeId);
    }

    void publish(int nodeId, PreparedStepMetadata metadata) {
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
