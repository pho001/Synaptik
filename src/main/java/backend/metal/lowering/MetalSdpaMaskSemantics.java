package backend.metal.lowering;

import graph.CompiledNode;
import graph.optimizer.partition.PartitionPlanningContext;
import operations.Operation;
import tensor.DataType;

import java.util.Arrays;

/**
 * Source-of-truth classifier for direct Metal SDPA public mask semantics.
 */
final class MetalSdpaMaskSemantics {
    private MetalSdpaMaskSemantics() {
    }

    static Decision classify(CompiledNode sdpaNode, PartitionPlanningContext context) {
        if (sdpaNode == null || context == null) {
            return Decision.unsupported(
                    MetalSdpaMaskMode.INVALID,
                    "UNSUPPORTED_RANK_OR_SHAPE",
                    "GPU_METAL SDPA mask semantics require planning context"
            );
        }
        if (sdpaNode.inputIds().size() == 3) {
            return Decision.supported(MetalSdpaMaskMode.UNMASKED);
        }
        if (sdpaNode.inputIds().size() != 4) {
            return Decision.unsupported(
                    MetalSdpaMaskMode.INVALID,
                    "UNSUPPORTED_RANK_OR_SHAPE",
                    "GPU_METAL masked SDPA requires query, key, value, and BOOL mask"
            );
        }

        CompiledNode query = context.compiledNode(sdpaNode.inputIds().get(0));
        CompiledNode key = context.compiledNode(sdpaNode.inputIds().get(1));
        CompiledNode mask = context.compiledNode(sdpaNode.inputIds().get(3));
        if (query == null || key == null || mask == null) {
            return Decision.unsupported(
                    MetalSdpaMaskMode.INVALID,
                    "UNSUPPORTED_RANK_OR_SHAPE",
                    "GPU_METAL SDPA mask inputs are unavailable"
            );
        }
        if (mask.dataType() != DataType.BOOL) {
            return Decision.unsupported(
                    MetalSdpaMaskMode.INVALID,
                    "UNSUPPORTED_DTYPE",
                    "GPU_METAL SDPA public mask input requires BOOL dtype"
            );
        }
        if (!mask.contiguous() || mask.hasStorageOffset()) {
            return Decision.unsupported(
                    classifyMaskMode(mask, context),
                    "UNSUPPORTED_LAYOUT",
                    "GPU_METAL SDPA mask input requires dense BOOL layout"
            );
        }
        int[] expectedScoresShape = expectedScoresShape(query.shape(), key.shape());
        if (expectedScoresShape.length == 0 || !Arrays.equals(mask.shape(), expectedScoresShape)) {
            return Decision.unsupported(
                    classifyMaskMode(mask, context),
                    "UNSUPPORTED_RANK_OR_SHAPE",
                    "GPU_METAL SDPA mask shape must equal broadcasted score shape"
            );
        }

        MetalSdpaMaskMode mode = classifyMaskMode(mask, context);
        return switch (mode) {
            case CAUSAL_BOOL_MASK -> Decision.unsupported(
                    mode,
                    "UNSUPPORTED_MASK_SEMANTICS",
                    "GPU_METAL SDPA mask mode CAUSAL_BOOL_MASK requires native causal mask support"
            );
            case EXTERNAL_AND_CAUSAL_BOOL_MASK -> Decision.unsupported(
                    mode,
                    "UNSUPPORTED_MASK_SEMANTICS",
                    "GPU_METAL SDPA mask mode EXTERNAL_AND_CAUSAL_BOOL_MASK requires native causal mask support"
            );
            case EXTERNAL_BOOL_MASK -> Decision.unsupported(
                    mode,
                    "UNSUPPORTED_MASK_SEMANTICS",
                    "GPU_METAL SDPA mask mode EXTERNAL_BOOL_MASK requires native BOOL mask ABI support"
            );
            default -> Decision.unsupported(
                    MetalSdpaMaskMode.INVALID,
                    "UNSUPPORTED_MASK_SEMANTICS",
                    "GPU_METAL SDPA mask mode is unsupported"
            );
        };
    }

    private static int[] expectedScoresShape(int[] queryShape, int[] keyShape) {
        if (queryShape == null || keyShape == null || queryShape.length < 2 || keyShape.length < 2) {
            return new int[0];
        }
        int[] qBatch = Arrays.copyOf(queryShape, queryShape.length - 2);
        int[] kBatch = Arrays.copyOf(keyShape, keyShape.length - 2);
        int[] batch = broadcastBatchShape(qBatch, kBatch);
        if (batch == null) {
            return new int[0];
        }
        int[] out = Arrays.copyOf(batch, batch.length + 2);
        out[out.length - 2] = queryShape[queryShape.length - 2];
        out[out.length - 1] = keyShape[keyShape.length - 2];
        return out;
    }

    private static int[] broadcastBatchShape(int[] left, int[] right) {
        int rank = Math.max(left.length, right.length);
        int[] out = new int[rank];
        for (int i = 0; i < rank; i++) {
            int leftIndex = left.length - 1 - i;
            int rightIndex = right.length - 1 - i;
            int l = leftIndex >= 0 ? left[leftIndex] : 1;
            int r = rightIndex >= 0 ? right[rightIndex] : 1;
            if (l != r && l != 1 && r != 1) {
                return null;
            }
            out[rank - 1 - i] = Math.max(l, r);
        }
        return out;
    }

    private static MetalSdpaMaskMode classifyMaskMode(CompiledNode mask, PartitionPlanningContext context) {
        CompiledNode source = unwrapExpand(mask, context);
        if (isCausalMaskLeaf(source)) {
            return MetalSdpaMaskMode.CAUSAL_BOOL_MASK;
        }
        if (source != null && source.operation() != null && source.operation().opType() == Operation.OpType.LOGICAL_AND) {
            boolean hasCausal = false;
            boolean hasExternal = false;
            for (int inputId : source.inputIds()) {
                CompiledNode input = unwrapExpand(context.compiledNode(inputId), context);
                if (isCausalMaskLeaf(input)) {
                    hasCausal = true;
                } else if (input != null) {
                    hasExternal = true;
                }
            }
            if (hasCausal && hasExternal) {
                return MetalSdpaMaskMode.EXTERNAL_AND_CAUSAL_BOOL_MASK;
            }
        }
        return MetalSdpaMaskMode.EXTERNAL_BOOL_MASK;
    }

    private static CompiledNode unwrapExpand(CompiledNode node, PartitionPlanningContext context) {
        CompiledNode current = node;
        while (current != null
                && current.operation() != null
                && current.operation().opType() == Operation.OpType.EXPAND
                && current.inputIds().size() == 1) {
            current = context.compiledNode(current.inputIds().getFirst());
        }
        return current;
    }

    private static boolean isCausalMaskLeaf(CompiledNode node) {
        return node != null
                && node.leaf()
                && node.dataType() == DataType.BOOL
                && "causal_mask".equals(node.label());
    }

    record Decision(MetalSdpaMaskMode mode, boolean supported, String reasonCode, String detail) {
        static Decision supported(MetalSdpaMaskMode mode) {
            return new Decision(mode, true, "SUPPORTED", "GPU_METAL SDPA mask mode " + mode + " is supported");
        }

        static Decision unsupported(MetalSdpaMaskMode mode, String reasonCode, String detail) {
            return new Decision(mode, false, reasonCode, detail);
        }

        String unsupportedReason() {
            return supported ? "" : reasonCode + ": " + detail;
        }
    }
}
