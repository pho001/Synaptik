# How-to guides

Choose a task once you know the [learning project's](../tutorials/index.md) basic [lifecycle](../glossary.md#lifecycle).
These guides answer practical questions; they do not replace the precise [API reference](../reference/index.md)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>.

The lifecycle separates describing a calculation, preparing reusable execution resources, running
with current data, and closing those resources. **[Gradients](../glossary.md#term-gradient)** describe how an [objective](../glossary.md#term-objective) changes;
an **[optimizer](../glossary.md#term-optimizer)** uses them to update [parameters](../glossary.md#term-parameter). **Accumulation** sums gradients from multiple
runs before an update. Use the [learning vocabulary and concrete batch](../tutorials/index.md#the-experiment-in-concrete-terms)<span data-glossary-navigation="true"> ([batch glossary](../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
when those tasks are new; the table below is navigation, not a prerequisite to understand every
advanced technique.

For high-level Model calls, your application supplies data while Model manages private copied
parameters and cached execution. For direct Engine calls, your application also manages each
run result and prepared handle. Choose [fresh-value binding](../user-guide/running-models.md#run-fresh-values-in-compiled-order)
when the next batch has new Tensor objects, and
[result materialization](../user-guide/running-models.md#run-and-materialize)
when output bytes must survive result closure. Neither workflow asks you to close every Tensor.

| I want to… | Guide |
|---|---|
| Train real numeric files, save a Model and see live terminal progress | [Interactive TUI and scriptable CLI](train-with-cli.md) |
| Choose [CPU](../glossary.md#term-cpu), [Metal](../glossary.md#term-metal) or explicit mixed execution | [Choose a backend deliberately](choose-a-backend.md) |
| Build an explicit Apple CPU numerical-library variant without changing backend ownership | [oneDNN with Accelerate](build-onednn-with-accelerate.md) |
| Save a trained [model](../glossary.md#term-model) and start fresh training after load | [Save and restart training](save-and-restart-training.md) |
| Find the stage responsible for a rejection or wrong expectation | [Diagnose a failed computation](diagnose-a-computation.md) |
| [Compile](../glossary.md#compile) and [prepare](../glossary.md#prepare) once, then [run](../glossary.md#term-run) fresh values | [Reusable prepared execution](../user-guide/running-models.md#compile-and-prepare-once)<span data-glossary-navigation="true"> ([prepared execution glossary](../glossary.md#prepared-execution--preparedexecution))</span> |
| Construct or transform [tensors](../glossary.md#tensor) | [Tensor tasks](../user-guide/tensors.md) |
| Build gradients without an optimizer update | [Automatic differentiation](../user-guide/autograd.md)<span data-glossary-navigation="true"> ([Automatic differentiation glossary](../glossary.md#autograd))</span> |
| Use optimizer groups, accumulation or in-memory state | [Training tasks](../user-guide/training.md) |
| Reconstruct a closed layer composition | [Declarative composition](../user-guide/declarative-models.md) |
| Prepare padded sequences or tokenize text | [Data reference](../reference/data/sequences.md) and [Text reference](../reference/text/tokenization.md) |

If you need the full picture rather than a procedure, read the [explanations](../explanation/index.md).
[Backend](../glossary.md#backend) authoring and [release](../glossary.md#term-release) mechanics belong in the [contributor area](../developer-guide/index.md).
