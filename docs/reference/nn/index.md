# Neural-network [Model](../../glossary.md#term-model) and layer contracts

Use a fluent [owned Model](../../glossary.md#model-module-buffer-and-scan-distinctions) for a fixed-[signature](../../glossary.md#term-signature) chain; use lower-level [modules](../../glossary.md#term-module) when you need to coordinate named state or custom composition. A Model owns its acquired [parameter](../../glossary.md#term-parameter) copies; a cold forward constructs expressions rather than running a [backend](../../glossary.md#backend).

[Reference goals](../index.md) · [Javadoc access](../javadoc.md)

A **layer** is a reusable neural-network building block; a **module** owns named parameters,
persistent [buffers](../../glossary.md#term-buffer) and children. **Parameters** are state selected for learning, while buffers
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
