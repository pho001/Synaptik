# Neural-network [Model](../../glossary.md#term-model) and layer contracts

Use a fluent [owned Model](../../glossary.md#model-module-buffer-and-scan-distinctions) for a fixed-[signature](../../glossary.md#term-signature) chain; use lower-level [modules](../../glossary.md#term-module) when you need to coordinate named state or custom composition. A Model owns its acquired [parameter](../../glossary.md#term-parameter) copies; a cold forward constructs expressions rather than running a [backend](../../glossary.md#backend).

[Reference goals](../index.md) · [Javadoc access](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc))</span>

A **layer** is a reusable neural-network building block; a **module** owns named parameters,
persistent [buffers](../../glossary.md#term-buffer) and children. **Parameters** are state selected for learning, while buffers
hold other persistent state. This section distinguishes ordinary expression-building modules from
owned Models that additionally [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) execution and privately acquired state.

## Choose a topic

- [Owned tensor Model execution](owned-model.md)<span data-glossary-navigation="true"> ([tensor glossary](../../glossary.md#tensor); [Model glossary](../../glossary.md#term-model))</span>
- [Runtime-length recurrent composition](recurrent.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime))</span>
- [Fluent owned Model builder](model-builder.md)<span data-glossary-navigation="true"> ([owned Model glossary](../../glossary.md#model-module-buffer-and-scan-distinctions))</span>
- [Typed Model composition and parameter initialization](composition.md)<span data-glossary-navigation="true"> ([parameter glossary](../../glossary.md#term-parameter))</span>
- [Channels-first convolution layers and examples](convolution.md)<span data-glossary-navigation="true"> ([convolution glossary](../../glossary.md#term-channels-first-convolution-layers))</span>
- [Unary and declarative composition](declarative.md)<span data-glossary-navigation="true"> ([Unary glossary](../../glossary.md#term-unary))</span>
- [RNN, GRU and LSTM layer contracts](recurrent-layers.md)<span data-glossary-navigation="true"> ([RNN glossary](../../glossary.md#term-recurrent-neural-network-rnn); [GRU glossary](../../glossary.md#term-gated-recurrent-unit-gru); [LSTM glossary](../../glossary.md#term-long-short-term-memory-lstm))</span>
- [Parameter updates, state dictionaries and Engine coordination](state.md)<span data-glossary-navigation="true"> ([Engine glossary](../../glossary.md#term-engine))</span>
