package graph.compile;

import backend.ComputeBackend;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;

/**
 * Structural contract for the user-visible tensor graph captured at compile time.
 *
 * <p>The contract deliberately excludes tensor storage contents and storage version counters so prepared executions
 * can be reused with new input values. It only verifies graph structure and metadata that prepared kernels rely on.</p>
 */
public final class GraphStructureContract {
    private static final GraphStructureContract UNCHECKED = new GraphStructureContract(List.of(), false);

    private final List<Node> nodes;
    private final boolean enforce;

    private GraphStructureContract(List<Node> nodes, boolean enforce) {
        this.nodes = List.copyOf(nodes == null ? List.of() : nodes);
        this.enforce = enforce;
    }

    public static GraphStructureContract capture(Tensor rootTensor) {
        Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        List<Tensor> graph = rootTensor.topologicalSort();
        IdentityHashMap<Tensor, Integer> ids = new IdentityHashMap<>();
        for (int i = 0; i < graph.size(); i++) {
            ids.put(graph.get(i), i);
        }

        ArrayList<Node> nodes = new ArrayList<>(graph.size());
        for (int i = 0; i < graph.size(); i++) {
            Tensor tensor = graph.get(i);
            List<Tensor> parents = tensor.getPrevTensors();
            int[] parentIds = new int[parents.size()];
            for (int p = 0; p < parents.size(); p++) {
                Integer parentId = ids.get(parents.get(p));
                if (parentId == null) {
                    throw new IllegalStateException("Graph contract parent is missing from topological graph.");
                }
                parentIds[p] = parentId;
            }
            Operation operation = tensor.getOperation();
            nodes.add(new Node(
                    tensor,
                    operation,
                    operation == null ? null : operation.opType(),
                    parentIds,
                    tensor.getShapeUnsafe(),
                    tensor.getDataType(),
                    tensor.resolveBackend()
            ));
        }
        return new GraphStructureContract(nodes, true);
    }

    public static GraphStructureContract unchecked() {
        return UNCHECKED;
    }

    public void validateOrThrow(Tensor rootTensor) {
        if (!enforce) {
            return;
        }
        Objects.requireNonNull(rootTensor, "rootTensor cannot be null");
        List<Tensor> current = rootTensor.topologicalSort();
        if (current.size() != nodes.size()) {
            throw stale("node count changed from " + nodes.size() + " to " + current.size());
        }
        IdentityHashMap<Tensor, Integer> ids = new IdentityHashMap<>();
        for (int i = 0; i < current.size(); i++) {
            ids.put(current.get(i), i);
        }
        for (int i = 0; i < nodes.size(); i++) {
            Node expected = nodes.get(i);
            Tensor actual = current.get(i);
            if (actual != expected.tensor()) {
                throw stale("node identity changed at index " + i);
            }
            Operation actualOperation = actual.getOperation();
            Operation.OpType actualOpType = actualOperation == null ? null : actualOperation.opType();
            if (actualOperation != expected.operation()) {
                throw stale("operation descriptor changed at index " + i);
            }
            if (actualOpType != expected.opType()) {
                throw stale("operation type changed at index " + i);
            }
            if (!Arrays.equals(actual.getShapeUnsafe(), expected.shape())) {
                throw stale("shape changed at index " + i);
            }
            if (actual.getDataType() != expected.dataType()) {
                throw stale("dtype changed at index " + i);
            }
            if (actual.resolveBackend() != expected.backend()) {
                throw stale("backend intent changed at index " + i);
            }
            List<Tensor> parents = actual.getPrevTensors();
            int[] expectedParentIds = expected.parentIds();
            if (parents.size() != expectedParentIds.length) {
                throw stale("input count changed at index " + i);
            }
            for (int p = 0; p < parents.size(); p++) {
                Integer parentId = ids.get(parents.get(p));
                if (parentId == null || parentId != expectedParentIds[p]) {
                    throw stale("input topology changed at index " + i);
                }
            }
        }
    }

    private static IllegalStateException stale(String reason) {
        return new IllegalStateException("Prepared execution graph contract is stale: " + reason + ".");
    }

    private record Node(
            Tensor tensor,
            Operation operation,
            Operation.OpType opType,
            int[] parentIds,
            int[] shape,
            DataType dataType,
            ComputeBackend backend
    ) {
        private Node {
            Objects.requireNonNull(tensor, "tensor cannot be null");
            parentIds = parentIds == null ? new int[0] : parentIds.clone();
            shape = shape == null ? new int[0] : shape.clone();
            Objects.requireNonNull(dataType, "dataType cannot be null");
            Objects.requireNonNull(backend, "backend cannot be null");
        }

        @Override
        public int[] parentIds() {
            return parentIds.clone();
        }

        @Override
        public int[] shape() {
            return shape.clone();
        }
    }
}
