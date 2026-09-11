package io.github.pho001.synaptik.prepare.analysis;

/**
 * Nominal role for one immutable selected-decision value owned by a concrete backend.
 *
 * <p>The role deliberately exposes no compatibility, matching, decoding, or application
 * operation. Shared Prepare code may retain and transport an implementation opaquely, while the
 * owning backend remains responsible for validating and applying its meaning.</p>
 */
public interface BackendTuningDecision { }
