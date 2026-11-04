# [Engine](../../glossary.md#term-engine): [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare) and [run](../../glossary.md#term-run)

Use ordinary Engine for reusable execution, or explicit composition when you must select opened [CPU](../../glossary.md#term-cpu)/[Metal](../../glossary.md#term-metal) [owners](../../glossary.md#term-owner). [Compilation](../../glossary.md#compiler-engine-and-execution-stage-names) retains logical recipes; prepared handles and result [leases](../../glossary.md#term-lease) [carry](../../glossary.md#term-carry) separate lifetimes.

[Reference goals](../index.md) · [Javadoc access](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc))</span>

Compile describes the computation, prepare constructs reusable execution recipes, and run
connects one set of stored inputs to those recipes. A result lease keeps that invocation's
completed resources alive; materializing a result makes an independent [host](../../glossary.md#term-host) copy. Reuse the
preparation for repeated runs, but close each result when its selected outputs are no longer needed.

## Choose a topic

- [CPU integration and native ownership](cpu-integration.md)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu); [ownership glossary](../../glossary.md#term-ownership-meanings))</span>
- [Ordinary Engine compile, prepare and run lifecycle](lifecycle.md)<span data-glossary-navigation="true"> ([Engine glossary](../../glossary.md#term-engine); [compile glossary](../../glossary.md#compile); [prepare glossary](../../glossary.md#prepare); [run glossary](../../glossary.md#term-run); [lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Advanced Engine lifecycle](advanced.md)
- [Explicit CPU and Metal composition](composition.md)<span data-glossary-navigation="true"> ([Metal glossary](../../glossary.md#term-metal))</span>
- [CPU limitations and development compatibility](limitations.md)
