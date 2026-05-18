package graph;

import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

/**
 * Central runtime aliasing rules for logical view operations.
 */
public final class AliasViewPolicy {
    private AliasViewPolicy() {
    }

    public static boolean aliasesInput0AtRuntime(Tensor tensor) {
        if (tensor == null || tensor.getOperation() == null) {
            return false;
        }
        List<Tensor> inputs = tensor.getPrevTensors();
        if (inputs == null || inputs.isEmpty()) {
            return false;
        }
        Operation.OpType opType = tensor.getOperation().opType();
        if (alwaysAliasesInput0AtRuntime(opType)) {
            return true;
        }
        return opType == Operation.OpType.RESHAPE && inputs.getFirst().isContiguous();
    }

    public static boolean aliasesInput0AtRuntime(
            CompiledNode node,
            CompiledTensorDescriptorIndex descriptorIndex
    ) {
        if (node == null || node.operation() == null || node.inputIds().isEmpty()) {
            return false;
        }
        Operation.OpType opType = node.operation().opType();
        if (alwaysAliasesInput0AtRuntime(opType)) {
            return true;
        }
        if (opType != Operation.OpType.RESHAPE) {
            return false;
        }
        CompiledTensorDescriptor source = descriptorIndex.byNodeId(node.inputIds().getFirst());
        return source.contiguous();
    }

    public static boolean alwaysAliasesInput0AtRuntime(Operation.OpType opType) {
        return switch (opType) {
            case NOOP, EXPAND, SELECT, SLICE, PERMUTE, EXPAND_DIMS, SQUEEZE -> true;
            default -> false;
        };
    }
}
