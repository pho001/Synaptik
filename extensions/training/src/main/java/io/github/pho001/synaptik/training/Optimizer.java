package io.github.pho001.synaptik.training;

/**
 * Marks one immutable built-in optimizer configuration accepted by {@link TrainingSession}.
 *
 * <p>An optimizer value contains configuration only. A session borrows that immutable value and
 * owns the step counter, gradient accumulation, and every per-parameter slot initialized from it.
 * Optimizers are backend-neutral: they expose no Runtime representation, kernel, device, or
 * concrete-backend option. The sealed set makes unsupported algorithms fail at compile time rather
 * than through a generic callback with unspecified state or atomicity.</p>
 */
public sealed interface Optimizer permits Sgd {
}
