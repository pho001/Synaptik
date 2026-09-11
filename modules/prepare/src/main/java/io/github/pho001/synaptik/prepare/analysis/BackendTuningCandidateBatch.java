package io.github.pho001.synaptik.prepare.analysis;

/**
 * Nominal role for one immutable, complete candidate batch owned by a concrete backend.
 *
 * <p>The role deliberately exposes no candidate collection, identity, schema, compatibility,
 * route, resource, measurement, cache, or persistence contract. Shared Prepare code may retain
 * and transport an implementation opaquely, but only its owning backend defines and interprets
 * the value.</p>
 */
public interface BackendTuningCandidateBatch { }
