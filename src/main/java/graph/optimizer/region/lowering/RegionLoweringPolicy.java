package graph.optimizer.region.lowering;

import graph.CompiledNode;

/**
 * Backend-aware policy for selecting the lowering form of a node inside an owned region.
 */
public interface RegionLoweringPolicy {
    RegionLoweringDecision decide(RegionLoweringPolicyContext context, CompiledNode node);
}
