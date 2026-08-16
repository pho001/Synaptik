package io.github.pho001.synaptik.nn.module;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.net.URI;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import javax.tools.DiagnosticCollector;
import javax.tools.JavaCompiler;
import javax.tools.JavaFileObject;
import javax.tools.SimpleJavaFileObject;
import javax.tools.StandardJavaFileManager;
import javax.tools.ToolProvider;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelTest {
    @Test
    void exposesExactlyThePlannedPublicAndPackageSurface() throws ReflectiveOperationException {
        Constructor<Model> modelConstructor = Model.class.getDeclaredConstructor();
        Method modelForward = Model.class.getDeclaredMethod("forward", Object.class);
        Method define = Model.class.getDeclaredMethod("define", ModelDefinition.class);
        Method definitionMethod = ModelDefinition.class.getDeclaredMethod("define", Topology.class);
        Method forwardMethod = ModelForward.class.getDeclaredMethod("forward", Object.class);
        Constructor<Topology> topologyConstructor = Topology.class.getDeclaredConstructor();
        Method addModule = Topology.class.getDeclaredMethod(
                "addModule", String.class, Module.class);
        Method namedRegistration = Module.class.getDeclaredMethod(
                "registerNamedChildren", Map.class);
        Set<String> topologyPublicMethods = Arrays.stream(Topology.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(Method::getName)
                .collect(Collectors.toSet());

        assertAll(
                () -> assertTrue(Modifier.isPublic(Model.class.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(Model.class.getModifiers())),
                () -> assertSame(Module.class, Model.class.getSuperclass()),
                () -> assertTrue(Modifier.isProtected(modelConstructor.getModifiers())),
                () -> assertTrue(Modifier.isPublic(modelForward.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(modelForward.getModifiers())),
                () -> assertTrue(Modifier.isPublic(define.getModifiers())),
                () -> assertTrue(Modifier.isStatic(define.getModifiers())),
                () -> assertEquals(Set.of("forward", "define"), Arrays.stream(Model.class.getDeclaredMethods())
                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                        .map(Method::getName)
                        .collect(Collectors.toSet())),
                () -> assertTrue(ModelDefinition.class.isAnnotationPresent(FunctionalInterface.class)),
                () -> assertTrue(ModelForward.class.isAnnotationPresent(FunctionalInterface.class)),
                () -> assertTrue(Modifier.isPublic(definitionMethod.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(definitionMethod.getModifiers())),
                () -> assertTrue(Modifier.isPublic(forwardMethod.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(forwardMethod.getModifiers())),
                () -> assertTrue(Modifier.isPublic(Topology.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(Topology.class.getModifiers())),
                () -> assertFalse(Modifier.isPublic(topologyConstructor.getModifiers())),
                () -> assertFalse(Modifier.isProtected(topologyConstructor.getModifiers())),
                () -> assertEquals(Set.of("addModule"), topologyPublicMethods),
                () -> assertTrue(Modifier.isPublic(addModule.getModifiers())),
                () -> assertFalse(Modifier.isPublic(namedRegistration.getModifiers())),
                () -> assertFalse(Modifier.isProtected(namedRegistration.getModifiers())),
                () -> assertTrue(Modifier.isFinal(namedRegistration.getModifiers())),
                () -> assertSame(void.class, namedRegistration.getReturnType()),
                () -> assertFalse(Arrays.stream(Model.class.getMethods())
                        .anyMatch(method -> method.getName().equals("backward"))));
    }

    @Test
    void externalJavaSourceInfersTensorAndRecordBoundariesWithVar(@TempDir Path output)
            throws IOException {
        String source = """
                package external;

                import io.github.pho001.synaptik.model.tensor.Tensor;
                import io.github.pho001.synaptik.nn.module.Model;
                import io.github.pho001.synaptik.nn.module.UnaryTensorModule;
                import java.util.Objects;

                public final class ModelInferenceProbe {
                    record Input(Tensor value) {}
                    record Output(Tensor value) {}

                    static Tensor tensorBoundary(Tensor input) {
                        var model = Model.define(topology -> {
                            var identity = topology.addModule("identity", new Identity());
                            return (Tensor value) -> identity.forward(value);
                        });
                        return model.forward(input);
                    }

                    static Output recordBoundary(Tensor input) {
                        var model = Model.define(topology ->
                                (Input value) -> new Output(value.value()));
                        return model.forward(new Input(input));
                    }

                    private static final class Identity extends UnaryTensorModule {
                        @Override
                        public Tensor forward(Tensor input) {
                            return Objects.requireNonNull(input, "input");
                        }
                    }
                }
                """;
        JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
        DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
        JavaFileObject compilationUnit = new SourceFile(
                "external.ModelInferenceProbe", source);

        try (StandardJavaFileManager files = compiler.getStandardFileManager(
                diagnostics, null, null)) {
            List<String> options = List.of(
                    "-proc:none",
                    "-classpath", System.getProperty("java.class.path"),
                    "-d", output.toString());
            boolean compiled = compiler.getTask(
                    null, files, diagnostics, options, null, List.of(compilationUnit)).call();
            assertTrue(compiled, () -> diagnostics.getDiagnostics().toString());
        }
    }

    @Test
    void preservesNamesDeclarationOrderExactChildrenAndDescriptiveStatePaths() {
        StateModule hidden = new StateModule("weight", "runningMean", 1.0d);
        StateModule output = new StateModule("bias", "runningVariance", 2.0d);

        var model = Model.define(topology -> {
            assertSame(hidden, topology.addModule("hidden", hidden));
            assertSame(output, topology.addModule("output", output));
            return (Tensor input) -> output.forward(hidden.forward(input));
        });

        assertAll(
                () -> assertEquals(List.of("hidden", "output"),
                        List.copyOf(model.children().keySet())),
                () -> assertSame(hidden, model.children().get("hidden")),
                () -> assertSame(output, model.children().get("output")),
                () -> assertEquals(List.of("hidden.weight", "output.bias"),
                        List.copyOf(model.parametersRecursively().keySet())),
                () -> assertEquals(List.of(
                                "hidden.runningMean", "output.runningVariance"),
                        List.copyOf(model.buffersRecursively().keySet())),
                () -> assertEquals(List.of(
                                "hidden.weight", "hidden.runningMean",
                                "output.bias", "output.runningVariance"),
                        model.stateDictionary().entries().stream()
                                .map(StateEntry::path)
                                .toList()));
    }

    @Test
    void permitsEmptyTopologyAndSealsCapturedTopologyAfterSuccess() {
        AtomicReference<Topology> captured = new AtomicReference<>();
        AtomicInteger definitionCalls = new AtomicInteger();
        Tensor input = tensor(false, "input");
        var model = Model.define(topology -> {
            definitionCalls.incrementAndGet();
            captured.set(topology);
            return (Tensor value) -> value;
        });

        Tensor result = model.forward(input);
        EmptyModule lateCandidate = new EmptyModule();
        IllegalStateException late = assertThrows(
                IllegalStateException.class,
                () -> captured.get().addModule("late", lateCandidate));
        var laterOwner = Model.define(topology -> {
            topology.addModule("late", lateCandidate);
            return (Object value) -> value;
        });

        assertAll(
                () -> assertSame(input, result),
                () -> assertEquals(1, definitionCalls.get()),
                () -> assertTrue(model.children().isEmpty()),
                () -> assertTrue(model.stateDictionary().entries().isEmpty()),
                () -> assertTrue(late.getMessage().contains("sealed")),
                () -> assertSame(lateCandidate, laterOwner.children().get("late")));
    }

    @Test
    void validatesDefinitionAndTopologyCollectionWithoutInstallingOwnership() {
        assertEquals("definition", assertThrows(
                NullPointerException.class,
                () -> Model.<Object, Object>define(null)).getMessage());

        AtomicReference<Topology> nullResultTopology = new AtomicReference<>();
        EmptyModule nullResultCandidate = new EmptyModule();
        NullPointerException nullResult = assertThrows(
                NullPointerException.class,
                () -> Model.<Object, Object>define(topology -> {
                    nullResultTopology.set(topology);
                    topology.addModule("candidate", nullResultCandidate);
                    return null;
                }));
        assertEquals("model forward", nullResult.getMessage());
        assertThrows(
                IllegalStateException.class,
                () -> nullResultTopology.get().addModule("late", new EmptyModule()));
        var nullResultCandidateOwner = Model.define(topology -> {
            topology.addModule("candidate", nullResultCandidate);
            return (Object value) -> value;
        });
        assertSame(nullResultCandidate, nullResultCandidateOwner.children().get("candidate"));

        for (String invalid : List.of("", " ", "nested.child")) {
            EmptyModule candidate = new EmptyModule();
            AtomicReference<Topology> captured = new AtomicReference<>();
            assertThrows(
                    IllegalArgumentException.class,
                    () -> Model.<Object, Object>define(topology -> {
                        captured.set(topology);
                        topology.addModule(invalid, candidate);
                        return value -> value;
                    }),
                    invalid);
            assertThrows(
                    IllegalStateException.class,
                    () -> captured.get().addModule("late", new EmptyModule()));
            assertTrue(candidate.children().isEmpty());
            Model.define(topology -> {
                topology.addModule("reused", candidate);
                return (Object value) -> value;
            });
        }

        EmptyModule nullNameCandidate = new EmptyModule();
        assertThrows(
                NullPointerException.class,
                () -> Model.<Object, Object>define(topology -> {
                    topology.addModule(null, nullNameCandidate);
                    return value -> value;
                }));
        assertTrue(nullNameCandidate.children().isEmpty());
        assertThrows(
                NullPointerException.class,
                () -> Model.<Object, Object>define(topology -> {
                    topology.addModule("module", null);
                    return value -> value;
                }));

        EmptyModule duplicateNameFirst = new EmptyModule();
        EmptyModule duplicateNameSecond = new EmptyModule();
        assertThrows(
                IllegalArgumentException.class,
                () -> Model.<Object, Object>define(topology -> {
                    topology.addModule("same", duplicateNameFirst);
                    topology.addModule("same", duplicateNameSecond);
                    return value -> value;
                }));
        Model.define(topology -> {
            topology.addModule("first", duplicateNameFirst);
            topology.addModule("second", duplicateNameSecond);
            return (Object value) -> value;
        });

        EmptyModule repeated = new EmptyModule();
        assertThrows(
                IllegalArgumentException.class,
                () -> Model.<Object, Object>define(topology -> {
                    topology.addModule("first", repeated);
                    topology.addModule("second", repeated);
                    return value -> value;
                }));
        Model.define(topology -> {
            topology.addModule("reused", repeated);
            return (Object value) -> value;
        });
    }

    @Test
    void callbackFailurePropagatesExactlySealsTopologyAndLeavesCandidatesReusable() {
        EmptyModule first = new EmptyModule();
        EmptyModule second = new EmptyModule();
        AtomicReference<Topology> captured = new AtomicReference<>();
        DeliberateFailure failure = new DeliberateFailure();

        DeliberateFailure thrown = assertThrows(
                DeliberateFailure.class,
                () -> Model.<Object, Object>define(topology -> {
                    captured.set(topology);
                    topology.addModule("first", first);
                    topology.addModule("second", second);
                    throw failure;
                }));

        assertSame(failure, thrown);
        assertThrows(
                IllegalStateException.class,
                () -> captured.get().addModule("late", new EmptyModule()));
        var owner = Model.define(topology -> {
            topology.addModule("first", first);
            topology.addModule("second", second);
            return (Object value) -> value;
        });
        assertEquals(List.of("first", "second"), List.copyOf(owner.children().keySet()));
    }

    @Test
    void completeOwnershipValidationPrecedesEveryInstallationAndSealsOnFailure() {
        EmptyModule alreadyOwned = new EmptyModule();
        NamedOwner existingOwner = new NamedOwner();
        existingOwner.attach("owned", alreadyOwned);
        EmptyModule validPrefix = new EmptyModule();
        AtomicReference<Topology> captured = new AtomicReference<>();

        assertThrows(
                IllegalStateException.class,
                () -> Model.<Object, Object>define(topology -> {
                    captured.set(topology);
                    topology.addModule("prefix", validPrefix);
                    topology.addModule("owned", alreadyOwned);
                    return value -> value;
                }));

        assertThrows(
                IllegalStateException.class,
                () -> captured.get().addModule("late", new EmptyModule()));
        var nextOwner = Model.define(topology -> {
            topology.addModule("prefix", validPrefix);
            return (Object value) -> value;
        });
        assertAll(
                () -> assertSame(alreadyOwned, existingOwner.children().get("owned")),
                () -> assertSame(validPrefix, nextOwner.children().get("prefix")));
    }

    @Test
    void namedRegistrationPreflightsNamesIdentityCyclesAndOwnershipAtomically() {
        NamedOwner receiver = new NamedOwner();
        receiver.declare("state");
        EmptyModule first = new EmptyModule();
        EmptyModule second = new EmptyModule();
        LinkedHashMap<String, Module> collision = new LinkedHashMap<>();
        collision.put("first", first);
        collision.put("state", second);

        assertThrows(IllegalArgumentException.class, () -> receiver.attachAll(collision));
        assertTrue(receiver.children().isEmpty());

        LinkedHashMap<String, Module> repeated = new LinkedHashMap<>();
        repeated.put("first", first);
        repeated.put("second", first);
        assertThrows(IllegalArgumentException.class, () -> receiver.attachAll(repeated));
        assertTrue(receiver.children().isEmpty());

        NamedOwner root = new NamedOwner();
        NamedOwner descendant = new NamedOwner();
        root.attach("descendant", descendant);
        LinkedHashMap<String, Module> cycle = new LinkedHashMap<>();
        cycle.put("prefix", first);
        cycle.put("ancestor", root);
        assertThrows(IllegalArgumentException.class, () -> descendant.attachAll(cycle));
        assertTrue(descendant.children().isEmpty());

        NamedOwner owner = new NamedOwner();
        owner.attach("owned", second);
        LinkedHashMap<String, Module> ownership = new LinkedHashMap<>();
        ownership.put("prefix", first);
        ownership.put("owned", second);
        assertThrows(IllegalStateException.class, () -> receiver.attachAll(ownership));
        assertTrue(receiver.children().isEmpty());

        NamedOwner finalOwner = new NamedOwner();
        finalOwner.attachAll(new LinkedHashMap<>(Map.of("first", first)));
        assertSame(first, finalOwner.children().get("first"));
    }

    @Test
    void functionalForwardValidatesReferencesAndPreservesPrefixFailureEffects() {
        AtomicInteger calls = new AtomicInteger();
        AtomicReference<Tensor> bodyInput = new AtomicReference<>();
        Tensor input = tensor(false, "input");
        Tensor output = tensor(false, "output");
        CountingModule unused = new CountingModule();
        var model = Model.define(topology -> {
            topology.addModule("unused", unused);
            return (Tensor value) -> {
                calls.incrementAndGet();
                bodyInput.set(value);
                return output;
            };
        });

        Tensor result = model.forward(input);
        NullPointerException nullInput = assertThrows(
                NullPointerException.class, () -> model.forward(null));

        assertAll(
                () -> assertSame(output, result),
                () -> assertSame(input, bodyInput.get()),
                () -> assertEquals(1, calls.get()),
                () -> assertEquals(0, unused.calls.get()),
                () -> assertEquals("input", nullInput.getMessage()));

        AtomicInteger nullCalls = new AtomicInteger();
        var nullModel = Model.define(topology -> (Tensor value) -> {
            nullCalls.incrementAndGet();
            return null;
        });
        NullPointerException nullOutput = assertThrows(
                NullPointerException.class, () -> nullModel.forward(input));
        assertAll(
                () -> assertEquals("model output", nullOutput.getMessage()),
                () -> assertEquals(1, nullCalls.get()));

        AtomicReference<Tensor> prefix = new AtomicReference<>();
        DeliberateFailure failure = new DeliberateFailure();
        var failingModel = Model.define(topology -> (Tensor value) -> {
            prefix.set(value.relu());
            throw failure;
        });
        DeliberateFailure thrown = assertThrows(
                DeliberateFailure.class, () -> failingModel.forward(input));
        assertAll(
                () -> assertSame(failure, thrown),
                () -> assertTrue(prefix.get().provenance().isPresent()));
    }

    @Test
    void supportsStructuredTypesAndInheritsModeReplacementAndStrictStateLoad() {
        StateModule module = new StateModule("weight", "runningMean", 1.0d);
        Model<StructuredInput, StructuredOutput> model = Model.define(topology -> {
            topology.addModule("encoder", module);
            return (StructuredInput input) -> new StructuredOutput(
                    module.forward(input.value()), input.tag());
        });
        Tensor input = tensor(false, "input");

        StructuredOutput output = model.forward(new StructuredInput(input, "sample"));
        Parameter parameter = model.parametersRecursively().get("encoder.weight");
        Tensor replacement = tensor(true, "replacement");
        parameter.replace(replacement);
        Tensor loadedParameter = tensor(true, "loaded-parameter");
        Tensor loadedBuffer = tensor(false, "loaded-buffer");
        model.eval();
        model.loadStateDictionary(new StateDictionary(List.of(
                new StateEntry("encoder.runningMean", StateKind.BUFFER, loadedBuffer),
                new StateEntry("encoder.weight", StateKind.PARAMETER, loadedParameter))));

        assertAll(
                () -> assertSame(input, output.value()),
                () -> assertEquals("sample", output.tag()),
                () -> assertSame(loadedParameter, parameter.value()),
                () -> assertSame(loadedBuffer,
                        model.buffersRecursively().get("encoder.runningMean").value()),
                () -> assertEquals(ForwardMode.EVALUATION, model.mode()),
                () -> assertEquals(ForwardMode.EVALUATION, module.mode()),
                () -> assertEquals(List.of("encoder.weight", "encoder.runningMean"),
                        model.stateDictionary().entries().stream().map(StateEntry::path).toList()));
    }

    private static Tensor tensor(boolean requiresGrad, String label) {
        return TensorFactory.zeros(
                Shape.scalar(), DataType.FLOAT64, Optional.of(label), requiresGrad);
    }

    private record StructuredInput(Tensor value, String tag) {
    }

    private record StructuredOutput(Tensor value, String tag) {
    }

    private static class EmptyModule extends Module {
    }

    private static final class CountingModule extends EmptyModule {
        private final AtomicInteger calls = new AtomicInteger();

        private Tensor forward(Tensor input) {
            calls.incrementAndGet();
            return input;
        }
    }

    private static final class StateModule extends Module {
        private final Parameter parameter;
        private final Buffer buffer;

        private StateModule(String parameterName, String bufferName, double value) {
            parameter = parameter(
                    parameterName,
                    TensorFactory.scalar(value, Optional.empty(), true));
            buffer = buffer(
                    bufferName,
                    TensorFactory.scalar(value, Optional.empty(), false));
        }

        private Tensor forward(Tensor input) {
            return input;
        }
    }

    private static final class NamedOwner extends Module {
        private void declare(String name) {
            buffer(name, tensor(false, name));
        }

        private <T extends Module> T attach(String name, T child) {
            return child(name, child);
        }

        private void attachAll(Map<String, ? extends Module> children) {
            registerNamedChildren(children);
        }
    }

    private static final class DeliberateFailure extends RuntimeException {
    }

    private static final class SourceFile extends SimpleJavaFileObject {
        private final String source;

        private SourceFile(String className, String source) {
            super(URI.create("string:///" + className.replace('.', '/') + Kind.SOURCE.extension),
                    Kind.SOURCE);
            this.source = source;
        }

        @Override
        public CharSequence getCharContent(boolean ignoreEncodingErrors) {
            return source;
        }
    }
}
