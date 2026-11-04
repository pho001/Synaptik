# Training: [parameter](../../glossary.md#term-parameter) updates and recovery

A [TrainingSession](../../glossary.md#training-session--trainingsession) borrows an [Engine](../../glossary.md#term-engine) and [live parameter source](../../glossary.md#live-parameter-source--trainingparameters), publishes [gradients](../../glossary.md#term-gradient), and [commits](../../glossary.md#commit-release-distribution-log-and-seed-meanings) one validated [update transaction](../../glossary.md#resource-transaction-and-rollback). [Model](../../glossary.md#term-model) convenience owns this [lifecycle](../../glossary.md#lifecycle) for its private state; [detached](../../glossary.md#detached-model-results) recovery is caller-coordinated.

[Reference goals](../index.md) · [Javadoc access](../javadoc.md)

Training uses [derivatives](../../glossary.md#term-derivative) to change parameter values. An **[optimizer](../../glossary.md#term-optimizer)** defines the update rule
and may remember earlier updates; a **[transaction](../../glossary.md#term-transaction)** validates a complete candidate before changing
the session's committed state. Recovery must distinguish that session state from model values and
from the application's next-input position.

A lower-level application calls `TrainingSession.open` and keeps its Engine and parameter
storage alive until session close. An owned NN Model makes those calls internally for its private
state. Start with [who calls whom during one step](sessions.md#who-calls-whom-during-one-step)
for the pre-update objective and external-batch lifetime; use
[private-candidate recovery](recovery.md) when model values, optimizer history and the next input
must be restored together.

## Choose a topic

- [Training sessions and owned Model convenience](sessions.md)
- [Parameter domains, admission and native copying](parameters.md)
- [Optimizers, gradient modes and atomic state](optimizers.md)
- [Caller-coordinated in-memory recovery](recovery.md)
