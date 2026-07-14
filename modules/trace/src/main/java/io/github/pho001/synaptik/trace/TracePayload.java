package io.github.pho001.synaptik.trace;

/**
 * Marks a typed diagnostic data-transfer object that can be carried by a trace event.
 *
 * <p>Implementations describe producer facts in trace-owned terms. They are not producer-domain
 * objects and do not perform traversal, execution, emission, or other business logic. Payload
 * implementations are required to be immutable, but this open method-free marker does not enforce
 * that property at runtime. The marker remains open so later typed payload families can be
 * introduced without a central registry.</p>
 */
public interface TracePayload {
}
