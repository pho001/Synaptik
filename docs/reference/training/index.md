# Training: [parameter](../../glossary.md#term-parameter) updates and recovery

A [TrainingSession](../../glossary.md#training-session--trainingsession) borrows an [Engine](../../glossary.md#term-engine) and [live parameter source](../../glossary.md#live-parameter-source--trainingparameters), publishes [gradients](../../glossary.md#term-gradient), and [commits](../../glossary.md#commit-release-distribution-log-and-seed-meanings) one validated [update transaction](../../glossary.md#resource-transaction-and-rollback). [Model](../../glossary.md#term-model) convenience owns this [lifecycle](../../glossary.md#lifecycle) for its private state; [detached](../../glossary.md#detached-model-results) recovery is caller-coordinated.

[Reference goals](../index.md) · [Javadoc access](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc))</span>

Training uses [derivatives](../../glossary.md#term-derivative) to change parameter values. An **[optimizer](../../glossary.md#term-optimizer)** defines the update rule
and may remember earlier updates; a **[transaction](../../glossary.md#term-transaction)** validates a complete candidate before changing
the session's committed state. Recovery must distinguish that session state from model values and
from the application's next-input position.

## Choose a topic

- [Training sessions and owned Model convenience](sessions.md)<span data-glossary-navigation="true"> ([owned Model glossary](../../glossary.md#model-module-buffer-and-scan-distinctions))</span>
- [Parameter domains, admission and native copying](parameters.md)<span data-glossary-navigation="true"> ([Parameter glossary](../../glossary.md#term-parameter); [admission glossary](../../glossary.md#term-admission))</span>
- [Optimizers, gradient modes and atomic state](optimizers.md)<span data-glossary-navigation="true"> ([Optimizers glossary](../../glossary.md#term-optimizer); [gradient glossary](../../glossary.md#term-gradient); [atomic glossary](../../glossary.md#resource-transaction-and-rollback))</span>
- [Caller-coordinated in-memory recovery](recovery.md)
