# How-to guides

Choose a task once you know the [learning project's](../tutorials/index.md) basic lifecycle.
These guides answer practical questions; they do not replace the precise [API reference](../reference/index.md).

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
