# Training: parameter updates and recovery

A TrainingSession borrows an Engine and live parameter source, publishes gradients, and commits one validated update transaction. Model convenience owns this lifecycle for its private state; detached recovery is caller-coordinated.

[Reference goals](../index.md) · [Javadoc access](../javadoc.md)

## Choose a topic

- [Training sessions and owned Model convenience](sessions.md)
- [Parameter domains, admission and native copying](parameters.md)
- [Optimizers, gradient modes and atomic state](optimizers.md)
- [Caller-coordinated in-memory recovery](recovery.md)
