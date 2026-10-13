package io.github.pho001.synaptik.training;

/**
 * Marks one immutable built-in optimizer configuration accepted by {@link TrainingSession}.
 *
 * <p>An optimizer value contains configuration only. A session borrows one default configuration
 * and may borrow further configurations through immutable {@link ParameterGroup} values. It owns
 * one global step counter and gradient accumulation lifecycle, a first slot for every parameter,
 * and a second slot only for Adam/AdamW assignments.
 * Optimizers are backend-neutral: they expose no Runtime representation, kernel, device, or
 * concrete-backend option. The sealed set makes unsupported algorithms fail at compile time rather
 * than through a generic callback with unspecified state or atomicity.</p>
 */
public sealed interface Optimizer permits Sgd, Adam, AdamW {
}
