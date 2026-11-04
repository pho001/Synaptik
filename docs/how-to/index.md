# How-to guides

Choose a task once you know the [learning project's](../tutorials/index.md) basic lifecycle.
These guides answer practical questions; they do not replace the precise [API reference](../reference/index.md).

The lifecycle separates describing a calculation, preparing reusable execution resources, running
with current data, and closing those resources. **Gradients** describe how an objective changes;
an **optimizer** uses them to update parameters. **Accumulation** sums gradients from multiple
runs before an update. Use the [learning vocabulary and concrete batch](../tutorials/index.md#the-experiment-in-concrete-terms)
when those tasks are new; the table below is navigation, not a prerequisite to understand every
advanced technique.

| I want to… | Guide |
|---|---|
| Choose CPU, Metal or explicit mixed execution | [Choose a backend deliberately](choose-a-backend.md) |
| Save a trained model and start fresh training after load | [Save and restart training](save-and-restart-training.md) |
| Find the stage responsible for a rejection or wrong expectation | [Diagnose a failed computation](diagnose-a-computation.md) |
| Compile and prepare once, then run fresh values | [Reusable prepared execution](../user-guide/running-models.md#compile-and-prepare-once) |
| Construct or transform tensors | [Tensor tasks](../user-guide/tensors.md) |
| Build gradients without an optimizer update | [Automatic differentiation](../user-guide/autograd.md) |
| Use optimizer groups, accumulation or in-memory state | [Training tasks](../user-guide/training.md) |
| Reconstruct a closed layer composition | [Declarative composition](../user-guide/declarative-models.md) |
| Prepare padded sequences or tokenize text | [Data reference](../reference/data/sequences.md) and [Text reference](../reference/text/tokenization.md) |

If you need the full picture rather than a procedure, read the [explanations](../explanation/index.md).
Backend authoring and release mechanics belong in the [contributor area](../developer-guide/index.md).
