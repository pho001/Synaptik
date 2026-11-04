package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.trace.TracePayload;

/**
 * Bounded native Scan validation context; contains no tensor values, handles or payload bytes.
 *
 * @param category native validation category: 1 bounds, 2 duplicate target, 3 BOOL domain
 * @param time original sequence time coordinate
 * @param unit compiled body unit ordinal, or -1 for a binder/publication role
 * @param operand input operand ordinal, or binder role ordinal when unit is -1
 * @param ordinal row-major logical failing index, update or BOOL element ordinal
 */
public record MetalScanValidationFailure(long category, long time, long unit,
        long operand, long ordinal) implements TracePayload {}
