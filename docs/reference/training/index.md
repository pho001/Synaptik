# Training: parameter updates and recovery

A TrainingSession borrows an Engine and live parameter source, publishes gradients, and commits one validated update transaction. Model convenience owns this lifecycle for its private state; detached recovery is caller-coordinated.

[Reference goals](../index.md) · [Javadoc access](../javadoc.md)

Training uses derivatives to change parameter values. An **optimizer** defines the update rule
and may remember earlier updates; a **transaction** validates a complete candidate before changing
the session's committed state. Recovery must distinguish that session state from model values and
from the application's next-input position.

## Choose a topic

- [Training sessions and owned Model convenience](sessions.md)
- [Parameter domains, admission and native copying](parameters.md)
- [Optimizers, gradient modes and atomic state](optimizers.md)
- [Caller-coordinated in-memory recovery](recovery.md)
