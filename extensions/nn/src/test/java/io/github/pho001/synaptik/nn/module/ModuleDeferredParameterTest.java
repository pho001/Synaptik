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
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class ModuleDeferredParameterTest {
    @Test
    void exposesExactlyTheFourProtectedFinalReservationPrimitives() throws Exception {
        Method reserve = Module.class.getDeclaredMethod(
                "reserveParameter", String.class, Consumer.class);
        Method bind = Module.class.getDeclaredMethod("bindReservedParameters", List.class);
        Method complete = Module.class.getDeclaredMethod("parameterReservationsBound");
        Method bound = Module.class.getDeclaredMethod("boundParameter", String.class);
        Set<String> names = Arrays.stream(Module.class.getDeclaredMethods())
                .filter(method -> Modifier.isProtected(method.getModifiers()))
                .map(Method::getName)
                .filter(name -> Set.of(
                                "reserveParameter",
                                "bindReservedParameters",
                                "parameterReservationsBound",
                                "boundParameter")
                        .contains(name))
                .collect(Collectors.toSet());

        assertAll(
                () -> assertEquals(
                        Set.of(
                                "reserveParameter",
                                "bindReservedParameters",
                                "parameterReservationsBound",
                                "boundParameter"),
                        names),
                () -> assertTrue(Modifier.isFinal(reserve.getModifiers())),
                () -> assertTrue(Modifier.isFinal(bind.getModifiers())),
                () -> assertTrue(Modifier.isFinal(complete.getModifiers())),
                () -> assertTrue(Modifier.isFinal(bound.getModifiers())),
                () -> assertFalse(Arrays.stream(Module.class.getMethods())
                        .anyMatch(method -> method.getName().contains("Reserv"))));
    }

    @Test
    void unboundReservationsFailClosedWhileBuffersChildrenAndModesRemainAvailable() {
        ReservedModule child = new ReservedModule();
        Owner root = new Owner(child);

        IllegalStateException direct = assertThrows(
                IllegalStateException.class, child::parameters);
        IllegalStateException recursive = assertThrows(
                IllegalStateException.class, root::parametersRecursively);
        IllegalStateException state = assertThrows(
                IllegalStateException.class, root::stateDictionary);

        assertAll(
                () -> assertTrue(direct.getMessage().contains("weight")),
                () -> assertTrue(recursive.getMessage().contains("layer.weight")),
                () -> assertTrue(state.getMessage().contains("layer.weight")),
                () -> assertEquals(List.of("running"), child.buffers().stream()
                        .map(Buffer::name).toList()),
                () -> assertSame(child, root.children().get("layer")));
        root.eval();
        assertEquals(ForwardMode.EVALUATION, child.mode());
    }

    @Test
    void bindingValidatesCompleteOrderedGroupBeforePublishingAndIsRetryable() {
        ReservedModule module = new ReservedModule();
        Tensor validWeight = parameter(Shape.of(4, 3));
        Tensor validBias = parameter(Shape.of(4));

        assertEquals("values", assertThrows(
                NullPointerException.class, () -> module.bind(null)).getMessage());
        assertThrows(IllegalArgumentException.class, () -> module.bind(List.of(validWeight)));
        assertThrows(
                NullPointerException.class,
                () -> module.bind(Arrays.asList(validWeight, null)));
        assertThrows(
                IllegalArgumentException.class,
                () -> module.bind(List.of(validWeight, parameter(Shape.of(5)))));
        assertFalse(module.complete());
        assertThrows(IllegalStateException.class, module::parameters);

        module.bind(List.of(validWeight, validBias));
        Parameter weight = module.get("weight");
        Parameter bias = module.get("bias");

        assertAll(
                () -> assertTrue(module.complete()),
                () -> assertSame(validWeight, weight.value()),
                () -> assertSame(validBias, bias.value()),
                () -> assertEquals(List.of(weight, bias), module.parameters()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> module.bind(List.of(validWeight, validBias))));
    }

    @Test
    void reservationsOccupyTheSharedNamespaceInEncounterOrder() {
        CollisionModule module = new CollisionModule();

        assertAll(
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> module.reserve("eager", value -> {})),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> module.declareBuffer("future")),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> module.attach("future", new EmptyModule())),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> module.declareParameter("future")),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> module.reserve("buffer", value -> {})),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> module.reserve("child", value -> {})));
    }

    @Test
    void reservationValidatesNameBeforeValidatorAndFailedDeclarationInstallsNothing() {
        EmptyReservableModule module = new EmptyReservableModule();

        assertAll(
                () -> assertEquals("name", assertThrows(
                                NullPointerException.class,
                                () -> module.reserve(null, null))
                        .getMessage()),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> module.reserve(" ", null)),
                () -> assertEquals("validator", assertThrows(
                                NullPointerException.class,
                                () -> module.reserve("future", null))
                        .getMessage()));

        module.reserve("future", value -> {});
        assertThrows(IllegalStateException.class, module::parameters);
    }

    private static Tensor parameter(Shape shape) {
        return TensorFactory.zeros(shape, DataType.FLOAT32, Optional.empty(), true);
    }

    private static class EmptyModule extends Module {
    }

    private static final class Owner extends Module {
        private Owner(Module child) {
            child("layer", child);
        }
    }

    private static final class ReservedModule extends Module {
        private ReservedModule() {
            reserveParameter("weight", value -> {
                if (!value.descriptor().shape().equals(Shape.of(4, 3))) {
                    throw new IllegalArgumentException("weight shape");
                }
            });
            reserveParameter("bias", value -> {
                if (!value.descriptor().shape().equals(Shape.of(4))) {
                    throw new IllegalArgumentException("bias shape");
                }
            });
            buffer("running", TensorFactory.scalar(0.0f, Optional.empty(), false));
        }

        private void bind(List<Tensor> values) {
            bindReservedParameters(values);
        }

        private boolean complete() {
            return parameterReservationsBound();
        }

        private Parameter get(String name) {
            return boundParameter(name);
        }
    }

    private static final class CollisionModule extends Module {
        private CollisionModule() {
            parameter("eager", ModuleDeferredParameterTest.parameter(Shape.of(1)));
            reserveParameter("future", value -> {});
            buffer("buffer", TensorFactory.scalar(0.0f, Optional.empty(), false));
            child("child", new EmptyModule());
        }

        private void reserve(String name, Consumer<Tensor> validator) {
            reserveParameter(name, validator);
        }

        private void declareBuffer(String name) {
            buffer(name, TensorFactory.scalar(0.0f, Optional.empty(), false));
        }

        private void declareParameter(String name) {
            parameter(name, ModuleDeferredParameterTest.parameter(Shape.of(1)));
        }

        private void attach(String name, Module child) {
            child(name, child);
        }
    }

    private static final class EmptyReservableModule extends Module {
        private void reserve(String name, Consumer<Tensor> validator) {
            reserveParameter(name, validator);
        }
    }
}
