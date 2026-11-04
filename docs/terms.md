# First terms

Use these entry points while reading the tutorial. The [complete glossary](glossary.md) remains the
single dictionary; this page chooses a reading order rather than creating different definitions.

| When you encounter… | Start with this distinction | Definition |
|---|---|---|
| Tensor | Typed, shaped values/expression, not an executing kernel | [Tensor](glossary.md#tensor) |
| Shape and layout | Index extents versus the mapping to storage | [Layout](glossary.md#layout), [worked geometry](explanation/tensors-and-gradients.md) |
| Batch and feature | Several samples versus the inputs describing one sample | [Learning data vocabulary](glossary.md#learning-data-vocabulary) |
| Parameter | Model state that an optimizer may change, not the next input batch | [Trainable parameter](glossary.md#trainable-parameter) |
| Prediction and loss | Output versus the chosen error objective | [Loss](glossary.md#loss) |
| Gradient and learning rate | Local sensitivity versus optimizer step scale | [Gradient and learning rate](glossary.md#gradient-and-learning-rate) |
| Automatic differentiation | Constructing chain-rule derivatives of supported expressions | [Compiler AD explanation](explanation/tensors-and-gradients.md#the-first-training-step) |
| Compile, prepare, run | Meaning/ownership, reusable realization, one invocation | [Stage comparison](glossary.md#compile-versus-prepare-versus-run) |
| Run result versus detached value | Live publication lease versus independent copied output | [Ownership](explanation/ownership.md) |
| Save versus resume | Source/current values versus optimizer and application progress | [Save and restart training](how-to/save-and-restart-training.md) |

Start with the [tutorial](tutorials/index.md) for a complete path. Use the
[reference](reference/index.md) when you know which operation or API boundary you need.
