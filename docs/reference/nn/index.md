# Neural-network Model and layer contracts

Use a fluent owned Model for a fixed-signature chain; use lower-level modules when you need to coordinate named state or custom composition. A Model owns its acquired parameter copies; a cold forward constructs expressions rather than running a backend.

[Reference goals](../index.md) · [Javadoc access](../javadoc.md)

A **layer** is a reusable neural-network building block; a **module** owns named parameters,
persistent buffers and children. **Parameters** are state selected for learning, while buffers
hold other persistent state. This section distinguishes ordinary expression-building modules from
owned Models that additionally coordinate execution and privately acquired state.

## Choose a topic

- [Owned tensor Model execution](owned-model.md)
- [Runtime-length recurrent composition](recurrent.md)
- [Fluent owned Model builder](model-builder.md)
- [Typed Model composition and parameter initialization](composition.md)
- [Channels-first convolution layers and examples](convolution.md)
- [Unary and declarative composition](declarative.md)
- [RNN, GRU and LSTM layer contracts](recurrent-layers.md)
- [Parameter updates, state dictionaries and Engine coordination](state.md)
