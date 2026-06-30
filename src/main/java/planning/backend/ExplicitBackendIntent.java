package planning.backend;

import config.compile.BackendTarget;

/**
 * One explicit accelerator backend intent attached to a compiled node.
 *
 * @param nodeId compiled graph node id carrying the intent
 * @param target requested accelerator target
 */
record ExplicitBackendIntent(int nodeId, BackendTarget target) {
    ExplicitBackendIntent {
        if (nodeId < 0) {
            throw new IllegalArgumentException("nodeId must be >= 0");
        }
        if (target == null || !target.accelerator()) {
            throw new IllegalArgumentException("target must be an accelerator backend");
        }
    }
}
