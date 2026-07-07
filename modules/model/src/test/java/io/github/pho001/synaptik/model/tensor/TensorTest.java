package io.github.pho001.synaptik.model.tensor;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.reduction.ArgMaxTiePolicy;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TensorTest {
    @Test
    void hasExactlyTheRequiredClassStateConstructorAndPublicApi()
            throws ReflectiveOperationException {
        assertAll(
                () -> assertTrue(Modifier.isPublic(Tensor.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(Tensor.class.getModifiers())),
                () -> assertFalse(Tensor.class.isRecord()),
                () -> assertEquals(Set.of(), Set.of(Tensor.class.getInterfaces())));

        var fields = Arrays.stream(Tensor.class.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();
        assertEquals(List.of("id", "descriptor", "label", "provenance", "hostStorage"),
                fields.stream().map(field -> field.getName()).toList());
        assertAll(
                () -> assertEquals(TensorId.class, fields.get(0).getType()),
                () -> assertEquals(TensorDescriptor.class, fields.get(1).getType()),
                () -> assertEquals(Optional.class, fields.get(2).getType()),
                () -> assertEquals(Optional.class, fields.get(3).getType()),
                () -> assertEquals(HostTensorStorage.class, fields.get(4).getType()),
                () -> assertTrue(fields.stream().allMatch(
                        field -> Modifier.isPrivate(field.getModifiers()))),
                () -> assertTrue(fields.subList(0, 4).stream().allMatch(
                        field -> Modifier.isFinal(field.getModifiers()))),
                () -> assertFalse(Modifier.isFinal(fields.get(4).getModifiers())));

        var constructors = Tensor.class.getDeclaredConstructors();
        assertEquals(1, constructors.length);
        assertAll(
                () -> assertFalse(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertFalse(Modifier.isProtected(constructors[0].getModifiers())),
                () -> assertFalse(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(
                        List.of(
                                TensorId.class,
                                TensorDescriptor.class,
                                Optional.class,
                                Optional.class,
                                Optional.class),
                        Arrays.asList(constructors[0].getParameterTypes())));

        var declaredPublicMethods = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .toList();
        Set<String> publicMethods = declaredPublicMethods.stream()
                .map(method -> method.getName())
                .collect(Collectors.toSet());
        assertEquals(102, declaredPublicMethods.size());
        assertEquals(
                Set.of("id", "descriptor", "label", "hostStorage", "replaceHostStorage",
                        "clearHostStorage", "provenance", "toString", "add", "sub", "mul",
                        "div", "min", "max", "pow", "abs", "neg", "inv", "log", "exp",
                        "erf", "sqrt", "floor", "ceil", "sign", "relu", "sigmoid", "tanh",
                        "fastExp", "fastTanh", "clamp", "clampMin", "clampMax", "greaterThan",
                        "greaterOrEqual", "lessThan", "lessOrEqual", "equalTo", "notEqualTo",
                        "logicalAnd", "logicalOr", "logicalNot", "where", "cast", "sum",
                        "mean", "prod", "all", "any", "argMax", "cumSum", "softmax",
                        "logSoftmax", "contiguous", "reshape", "expand", "permute",
                        "transpose", "expandDims", "squeeze", "slice", "sliceAxis", "select",
                        "gather", "gatherAxis", "take", "takeAlongAxis", "pad",
                        "tile", "concat", "stack", "unstack", "unfold", "foldAxis",
                        "unfold2d", "fold2d"),
                publicMethods);
        assertAll(
                () -> assertTrue(Modifier.isSynchronized(
                        Tensor.class.getDeclaredMethod("hostStorage").getModifiers())),
                () -> assertTrue(Modifier.isSynchronized(
                        Tensor.class.getDeclaredMethod(
                                        "replaceHostStorage", HostTensorStorage.class)
                                .getModifiers())),
                () -> assertTrue(Modifier.isSynchronized(
                        Tensor.class.getDeclaredMethod("clearHostStorage").getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(
                        Tensor.class.getDeclaredMethod("id").getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(
                        Tensor.class.getDeclaredMethod("descriptor").getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(
                        Tensor.class.getDeclaredMethod("label").getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(
                        Tensor.class.getDeclaredMethod("provenance").getModifiers())));

        for (String methodName : List.of("add", "sub", "mul", "div", "min", "max", "pow")) {
            var method = Tensor.class.getDeclaredMethod(methodName, Tensor.class);
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }

        for (String methodName : List.of(
                "greaterThan", "greaterOrEqual", "lessThan", "lessOrEqual", "equalTo",
                "notEqualTo", "logicalAnd", "logicalOr")) {
            var method = Tensor.class.getDeclaredMethod(methodName, Tensor.class);
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertEquals(List.of(Tensor.class),
                            Arrays.asList(method.getParameterTypes())),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }

        var logicalNot = Tensor.class.getDeclaredMethod("logicalNot");
        assertAll(
                () -> assertEquals(Tensor.class, logicalNot.getReturnType()),
                () -> assertEquals(0, logicalNot.getParameterCount()),
                () -> assertTrue(Modifier.isPublic(logicalNot.getModifiers())),
                () -> assertFalse(Modifier.isStatic(logicalNot.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(logicalNot.getModifiers())));

        var where = Tensor.class.getDeclaredMethod(
                "where", Tensor.class, Tensor.class, Tensor.class);
        assertAll(
                () -> assertEquals(Tensor.class, where.getReturnType()),
                () -> assertEquals(
                        List.of(Tensor.class, Tensor.class, Tensor.class),
                        Arrays.asList(where.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(where.getModifiers())),
                () -> assertTrue(Modifier.isStatic(where.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(where.getModifiers())));

        var cast = Tensor.class.getDeclaredMethod("cast", DataType.class);
        assertAll(
                () -> assertEquals(Tensor.class, cast.getReturnType()),
                () -> assertEquals(
                        List.of(DataType.class), Arrays.asList(cast.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(cast.getModifiers())),
                () -> assertFalse(Modifier.isStatic(cast.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(cast.getModifiers())));

        for (String methodName : List.of(
                "sum", "mean", "prod", "min", "max", "all", "any")) {
            var full = Tensor.class.getDeclaredMethod(methodName);
            var axis = Tensor.class.getDeclaredMethod(methodName, int.class);
            var retained = Tensor.class.getDeclaredMethod(
                    methodName, int.class, boolean.class);
            for (var method : List.of(full, axis, retained)) {
                assertAll(
                        () -> assertEquals(Tensor.class, method.getReturnType()),
                        () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                        () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                        () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
            }
            assertAll(
                    () -> assertEquals(0, full.getParameterCount()),
                    () -> assertEquals(
                            List.of(int.class), Arrays.asList(axis.getParameterTypes())),
                    () -> assertEquals(
                            List.of(int.class, boolean.class),
                            Arrays.asList(retained.getParameterTypes())));
        }

        for (String methodName : List.of("sum", "mean")) {
            var masked = Tensor.class.getDeclaredMethod(
                    methodName, int.class, Tensor.class);
            assertAll(
                    () -> assertEquals(Tensor.class, masked.getReturnType()),
                    () -> assertEquals(
                            List.of(int.class, Tensor.class),
                            Arrays.asList(masked.getParameterTypes())),
                    () -> assertTrue(Modifier.isPublic(masked.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(masked.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(masked.getModifiers())));
        }

        assertAll(
                () -> assertEquals(
                        Tensor.class,
                        Tensor.class.getDeclaredMethod("argMax", int.class).getReturnType()),
                () -> assertEquals(
                        Tensor.class,
                        Tensor.class.getDeclaredMethod("argMax", int.class, boolean.class)
                                .getReturnType()),
                () -> assertEquals(
                        Tensor.class,
                        Tensor.class.getDeclaredMethod(
                                        "argMax", int.class, boolean.class, ArgMaxTiePolicy.class)
                                .getReturnType()));

        for (Class<?>[] parameters : List.of(
                new Class<?>[] {int.class},
                new Class<?>[] {int.class, boolean.class, boolean.class})) {
            var method = Tensor.class.getDeclaredMethod("cumSum", parameters);
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertEquals(List.of(parameters),
                            Arrays.asList(method.getParameterTypes())),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }

        for (String methodName : List.of("softmax", "logSoftmax")) {
            var method = Tensor.class.getDeclaredMethod(methodName, int.class);
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertEquals(
                            List.of(int.class), Arrays.asList(method.getParameterTypes())),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }

        var contiguous = Tensor.class.getDeclaredMethod("contiguous");
        assertAll(
                () -> assertEquals(Tensor.class, contiguous.getReturnType()),
                () -> assertEquals(0, contiguous.getParameterCount()),
                () -> assertTrue(Modifier.isPublic(contiguous.getModifiers())),
                () -> assertFalse(Modifier.isStatic(contiguous.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(contiguous.getModifiers())));

        for (Class<?> parameter : List.of(long[].class, Shape.class)) {
            var reshape = Tensor.class.getDeclaredMethod("reshape", parameter);
            assertAll(
                    () -> assertEquals(Tensor.class, reshape.getReturnType()),
                    () -> assertEquals(List.of(parameter),
                            Arrays.asList(reshape.getParameterTypes())),
                    () -> assertTrue(Modifier.isPublic(reshape.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(reshape.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(reshape.getModifiers())),
                    () -> assertEquals(parameter == long[].class, reshape.isVarArgs()));

            var expand = Tensor.class.getDeclaredMethod("expand", parameter);
            assertAll(
                    () -> assertEquals(Tensor.class, expand.getReturnType()),
                    () -> assertEquals(List.of(parameter),
                            Arrays.asList(expand.getParameterTypes())),
                    () -> assertTrue(Modifier.isPublic(expand.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(expand.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(expand.getModifiers())),
                    () -> assertEquals(parameter == long[].class, expand.isVarArgs()));
        }

        var permute = Tensor.class.getDeclaredMethod("permute", int[].class);
        var transpose = Tensor.class.getDeclaredMethod("transpose");
        var expandDims = Tensor.class.getDeclaredMethod("expandDims", int.class);
        var squeeze = Tensor.class.getDeclaredMethod("squeeze", int.class);
        assertAll(
                () -> assertEquals(Tensor.class, permute.getReturnType()),
                () -> assertEquals(List.of(int[].class),
                        Arrays.asList(permute.getParameterTypes())),
                () -> assertTrue(permute.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(permute.getModifiers())),
                () -> assertFalse(Modifier.isStatic(permute.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(permute.getModifiers())),
                () -> assertEquals(Tensor.class, transpose.getReturnType()),
                () -> assertEquals(0, transpose.getParameterCount()),
                () -> assertFalse(transpose.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(transpose.getModifiers())),
                () -> assertFalse(Modifier.isStatic(transpose.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(transpose.getModifiers())),
                () -> assertEquals(Tensor.class, expandDims.getReturnType()),
                () -> assertEquals(List.of(int.class),
                        Arrays.asList(expandDims.getParameterTypes())),
                () -> assertFalse(expandDims.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(expandDims.getModifiers())),
                () -> assertFalse(Modifier.isStatic(expandDims.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(expandDims.getModifiers())),
                () -> assertEquals(Tensor.class, squeeze.getReturnType()),
                () -> assertEquals(List.of(int.class),
                        Arrays.asList(squeeze.getParameterTypes())),
                () -> assertFalse(squeeze.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(squeeze.getModifiers())),
                () -> assertFalse(Modifier.isStatic(squeeze.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(squeeze.getModifiers())));

        var slice = Tensor.class.getDeclaredMethod(
                "slice", long[].class, long[].class, int[].class, long[].class);
        var sliceAxis = Tensor.class.getDeclaredMethod(
                "sliceAxis", int.class, long.class, long.class);
        assertAll(
                () -> assertEquals(Tensor.class, slice.getReturnType()),
                () -> assertEquals(
                        List.of(long[].class, long[].class, int[].class, long[].class),
                        Arrays.asList(slice.getParameterTypes())),
                () -> assertFalse(slice.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(slice.getModifiers())),
                () -> assertFalse(Modifier.isStatic(slice.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(slice.getModifiers())),
                () -> assertEquals(Tensor.class, sliceAxis.getReturnType()),
                () -> assertEquals(
                        List.of(int.class, long.class, long.class),
                        Arrays.asList(sliceAxis.getParameterTypes())),
                () -> assertFalse(sliceAxis.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(sliceAxis.getModifiers())),
                () -> assertFalse(Modifier.isStatic(sliceAxis.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(sliceAxis.getModifiers())));

        var select = Tensor.class.getDeclaredMethod("select", int.class, long.class);
        assertAll(
                () -> assertEquals(Tensor.class, select.getReturnType()),
                () -> assertEquals(
                        List.of(int.class, long.class),
                        Arrays.asList(select.getParameterTypes())),
                () -> assertFalse(select.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(select.getModifiers())),
                () -> assertFalse(Modifier.isStatic(select.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(select.getModifiers())));

        for (String methodName : List.of("gather", "gatherAxis", "takeAlongAxis")) {
            var method = Tensor.class.getDeclaredMethod(
                    methodName, Tensor.class, int.class);
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertEquals(
                            List.of(Tensor.class, int.class),
                            Arrays.asList(method.getParameterTypes())),
                    () -> assertFalse(method.isVarArgs()),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }

        for (Class<?> indexType : List.of(Tensor.class, int[].class)) {
            var take = Tensor.class.getDeclaredMethod("take", int.class, indexType);
            assertAll(
                    () -> assertEquals(Tensor.class, take.getReturnType()),
                    () -> assertEquals(
                            List.of(int.class, indexType),
                            Arrays.asList(take.getParameterTypes())),
                    () -> assertFalse(take.isVarArgs()),
                    () -> assertTrue(Modifier.isPublic(take.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(take.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(take.getModifiers())));
        }

        var pad = Tensor.class.getDeclaredMethod(
                "pad", long[].class, long[].class, double.class);
        var tile = Tensor.class.getDeclaredMethod("tile", long[].class);
        assertAll(
                () -> assertEquals(Tensor.class, pad.getReturnType()),
                () -> assertEquals(
                        List.of(long[].class, long[].class, double.class),
                        Arrays.asList(pad.getParameterTypes())),
                () -> assertFalse(pad.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(pad.getModifiers())),
                () -> assertFalse(Modifier.isStatic(pad.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(pad.getModifiers())),
                () -> assertEquals(Tensor.class, tile.getReturnType()),
                () -> assertEquals(
                        List.of(long[].class), Arrays.asList(tile.getParameterTypes())),
                () -> assertTrue(tile.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(tile.getModifiers())),
                () -> assertFalse(Modifier.isStatic(tile.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(tile.getModifiers())));

        var concat = Tensor.class.getDeclaredMethod("concat", int.class, Tensor[].class);
        var stack = Tensor.class.getDeclaredMethod("stack", int.class, Tensor[].class);
        var unstack = Tensor.class.getDeclaredMethod("unstack", int.class);
        assertAll(
                () -> assertEquals(Tensor.class, concat.getReturnType()),
                () -> assertEquals(
                        List.of(int.class, Tensor[].class),
                        Arrays.asList(concat.getParameterTypes())),
                () -> assertTrue(concat.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(concat.getModifiers())),
                () -> assertTrue(Modifier.isStatic(concat.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(concat.getModifiers())),
                () -> assertEquals(Tensor.class, stack.getReturnType()),
                () -> assertEquals(
                        List.of(int.class, Tensor[].class),
                        Arrays.asList(stack.getParameterTypes())),
                () -> assertTrue(stack.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(stack.getModifiers())),
                () -> assertTrue(Modifier.isStatic(stack.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(stack.getModifiers())),
                () -> assertEquals(List.class, unstack.getReturnType()),
                () -> assertEquals(
                        List.of(int.class), Arrays.asList(unstack.getParameterTypes())),
                () -> assertFalse(unstack.isVarArgs()),
                () -> assertTrue(Modifier.isPublic(unstack.getModifiers())),
                () -> assertFalse(Modifier.isStatic(unstack.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(unstack.getModifiers())));

        for (var method : List.of(
                Tensor.class.getDeclaredMethod(
                        "unfold", int.class, long.class, long.class),
                Tensor.class.getDeclaredMethod(
                        "foldAxis", int.class, long.class, long.class),
                Tensor.class.getDeclaredMethod("unfold2d", Window2dAttrs.class),
                Tensor.class.getDeclaredMethod(
                        "fold2d", Shape.class, Window2dAttrs.class))) {
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())),
                    () -> assertFalse(method.isVarArgs()));
        }

        for (String methodName : List.of("mul", "pow", "clampMin", "clampMax")) {
            var method = Tensor.class.getDeclaredMethod(methodName, double.class);
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertEquals(List.of(double.class),
                            Arrays.asList(method.getParameterTypes())),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }

        var clamp = Tensor.class.getDeclaredMethod("clamp", double.class, double.class);
        assertAll(
                () -> assertEquals(Tensor.class, clamp.getReturnType()),
                () -> assertEquals(List.of(double.class, double.class),
                        Arrays.asList(clamp.getParameterTypes())),
                () -> assertTrue(Modifier.isPublic(clamp.getModifiers())),
                () -> assertFalse(Modifier.isStatic(clamp.getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(clamp.getModifiers())));

        for (String methodName : List.of(
                "abs", "neg", "inv", "log", "exp", "erf", "sqrt", "floor", "ceil", "sign",
                "relu", "sigmoid", "tanh", "fastExp", "fastTanh")) {
            var method = Tensor.class.getDeclaredMethod(methodName);
            assertAll(
                    () -> assertEquals(Tensor.class, method.getReturnType()),
                    () -> assertEquals(0, method.getParameterCount()),
                    () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                    () -> assertFalse(Modifier.isStatic(method.getModifiers())),
                    () -> assertFalse(Modifier.isSynchronized(method.getModifiers())));
        }
    }

    @Test
    void validatesConstructorReferencesAndBlankLabelInDeterministicOrder() {
        TensorDescriptor descriptor = unresolved(DataType.FLOAT32, Shape.of(2, 3));
        HostTensorStorage wrongType = storage(DataType.INT32, 0);

        NullPointerException nullId = assertThrows(
                NullPointerException.class,
                () -> new Tensor(null, null, null, null, null));
        NullPointerException nullDescriptor = assertThrows(
                NullPointerException.class,
                () -> new Tensor(new TensorId(1), null, null, null, null));
        NullPointerException nullLabel = assertThrows(
                NullPointerException.class,
                () -> new Tensor(new TensorId(1), descriptor, null, null, null));
        NullPointerException nullProvenance = assertThrows(
                NullPointerException.class,
                () -> new Tensor(new TensorId(1), descriptor, Optional.empty(), null, null));
        NullPointerException nullStorageOptional = assertThrows(
                NullPointerException.class,
                () -> new Tensor(
                        new TensorId(1),
                        descriptor,
                        Optional.of(" "),
                        Optional.empty(),
                        null));
        IllegalArgumentException blankWins = assertThrows(
                IllegalArgumentException.class,
                () -> new Tensor(
                        new TensorId(1),
                        descriptor,
                        Optional.of(" \t\n "),
                        Optional.empty(),
                        Optional.of(wrongType)));

        assertAll(
                () -> assertEquals("id", nullId.getMessage()),
                () -> assertEquals("descriptor", nullDescriptor.getMessage()),
                () -> assertEquals("label", nullLabel.getMessage()),
                () -> assertEquals("provenance", nullProvenance.getMessage()),
                () -> assertEquals("hostStorage", nullStorageOptional.getMessage()),
                () -> assertEquals("label must not be blank", blankWins.getMessage()));
    }

    @Test
    void retainsStableReferencesAndNormalizesLabelValue() {
        TensorId id = new TensorId(7);
        TensorDescriptor descriptor = unresolved(DataType.FLOAT32, Shape.of(2, 3));
        Tensor labeled = new Tensor(
                id, descriptor, Optional.of("  weights\n"), Optional.empty(), Optional.empty());
        Tensor unlabeled = new Tensor(
                new TensorId(8),
                descriptor,
                Optional.empty(),
                Optional.empty(),
                Optional.empty());

        assertAll(
                () -> assertSame(id, labeled.id()),
                () -> assertSame(descriptor, labeled.descriptor()),
                () -> assertEquals(Optional.of("weights"), labeled.label()),
                () -> assertEquals(Optional.empty(), unlabeled.label()),
                () -> assertEquals(Optional.empty(), labeled.provenance()),
                () -> assertEquals(Optional.empty(), labeled.hostStorage()));
    }

    @Test
    void retainsExactImmutableProvenanceIndependentlyOfHostStorage() {
        TensorDescriptor descriptor = unresolved(DataType.FLOAT32, Shape.scalar());
        Arena arena = Arena.ofConfined();
        HostTensorStorage inputStorage = new MemorySegmentStorage(
                DataType.FLOAT32, 0, arena.allocate(0, 1));
        Tensor input = new Tensor(
                new TensorId(6),
                descriptor,
                Optional.empty(),
                Optional.empty(),
                Optional.of(inputStorage));
        TensorProvenance provenance = new TensorProvenance(
                new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE), List.of(input));
        Tensor derived = new Tensor(
                new TensorId(7),
                descriptor,
                Optional.of("derived"),
                Optional.of(provenance),
                Optional.empty());
        String initialText = derived.toString();

        HostTensorStorage derivedStorage = storage(DataType.FLOAT32, 0);
        derived.replaceHostStorage(derivedStorage);
        arena.close();
        Optional<HostTensorStorage> clearedInput = input.clearHostStorage();
        Optional<HostTensorStorage> clearedDerived = derived.clearHostStorage();

        assertAll(
                () -> assertSame(provenance, derived.provenance().orElseThrow()),
                () -> assertSame(input, derived.provenance().orElseThrow().inputs().getFirst()),
                () -> assertSame(inputStorage, clearedInput.orElseThrow()),
                () -> assertFalse(inputStorage.isAlive()),
                () -> assertSame(derivedStorage, clearedDerived.orElseThrow()),
                () -> assertEquals(initialText, derived.toString()),
                () -> assertFalse(initialText.contains("provenance")),
                () -> assertFalse(initialText.contains("operation")),
                () -> assertFalse(initialText.contains("SAMPLE")));
    }

    @Test
    void validatesStorageTypeCapacityAndLivenessInExactOrder() {
        TensorDescriptor resolved = resolved(
                DataType.FLOAT32,
                Shape.of(2, 3),
                LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 2, true));
        Arena arena = Arena.ofConfined();
        MemorySegment deadEightBytes = arena.allocate(8, 1);
        MemorySegmentStorage wrongDead =
                new MemorySegmentStorage(DataType.INT64, 1, deadEightBytes);
        MemorySegmentStorage smallDead = new MemorySegmentStorage(
                DataType.FLOAT32, 2, deadEightBytes);
        arena.close();

        IllegalArgumentException typeWins = assertThrows(
                IllegalArgumentException.class,
                () -> new Tensor(
                        new TensorId(1),
                        resolved,
                        Optional.empty(),
                        Optional.empty(),
                        Optional.of(wrongDead)));
        IllegalArgumentException capacityWins = assertThrows(
                IllegalArgumentException.class,
                () -> new Tensor(
                        new TensorId(1),
                        resolved,
                        Optional.empty(),
                        Optional.empty(),
                        Optional.of(smallDead)));
        TensorDescriptor unresolved = unresolved(DataType.FLOAT32, Shape.of(2, 3));
        IllegalStateException liveness = assertThrows(
                IllegalStateException.class,
                () -> new Tensor(
                        new TensorId(1),
                        unresolved,
                        Optional.empty(),
                        Optional.empty(),
                        Optional.of(smallDead)));

        assertAll(
                () -> assertEquals(
                        "hostStorage data type must match descriptor data type: expected=FLOAT32, actual=INT64",
                        typeWins.getMessage()),
                () -> assertEquals(
                        "hostStorage element capacity is smaller than resolved layout span: required=8, actual=2",
                        capacityWins.getMessage()),
                () -> assertEquals("hostStorage must be alive when attached", liveness.getMessage()));
    }

    @Test
    void replacementIsAtomicAndReturnsExactPreviousReferences() {
        HostTensorStorage first = storage(DataType.FLOAT32, 6);
        HostTensorStorage second = storage(DataType.FLOAT32, 8);
        HostTensorStorage invalidType = storage(DataType.INT32, 6);
        Tensor initiallyEmpty = new Tensor(
                new TensorId(0), unresolved(DataType.FLOAT32, Shape.of(2, 3)), Optional.empty(),
                Optional.empty(),
                Optional.empty());
        Tensor tensor = new Tensor(
                new TensorId(1),
                resolved(DataType.FLOAT32, Shape.of(2, 3), LayoutDescriptor.contiguous(Shape.of(2, 3))),
                Optional.empty(),
                Optional.empty(),
                Optional.of(first));
        Optional<HostTensorStorage> initialSnapshot = tensor.hostStorage();

        Optional<HostTensorStorage> absentPrevious = initiallyEmpty.replaceHostStorage(first);
        Optional<HostTensorStorage> replaced = tensor.replaceHostStorage(second);
        IllegalArgumentException failed = assertThrows(
                IllegalArgumentException.class,
                () -> tensor.replaceHostStorage(invalidType));
        Optional<HostTensorStorage> cleared = tensor.clearHostStorage();
        Optional<HostTensorStorage> clearedAgain = tensor.clearHostStorage();

        assertAll(
                () -> assertEquals(Optional.empty(), absentPrevious),
                () -> assertSame(first, initiallyEmpty.hostStorage().orElseThrow()),
                () -> assertSame(first, initialSnapshot.orElseThrow()),
                () -> assertSame(first, replaced.orElseThrow()),
                () -> assertSame(second, cleared.orElseThrow()),
                () -> assertEquals(Optional.empty(), clearedAgain),
                () -> assertEquals(Optional.empty(), tensor.hostStorage()),
                () -> assertSame(first, initialSnapshot.orElseThrow()),
                () -> assertEquals(
                        "hostStorage data type must match descriptor data type: expected=FLOAT32, actual=INT32",
                        failed.getMessage()));
    }

    @Test
    void replacementRejectsNullWithoutChangingAssociation() {
        HostTensorStorage first = storage(DataType.FLOAT32, 0);
        Tensor tensor = new Tensor(
                new TensorId(1), unresolved(DataType.FLOAT32, Shape.of(2, 3)), Optional.empty(),
                Optional.empty(),
                Optional.of(first));

        NullPointerException failure = assertThrows(
                NullPointerException.class, () -> tensor.replaceHostStorage(null));

        assertAll(
                () -> assertEquals("hostStorage", failure.getMessage()),
                () -> assertSame(first, tensor.hostStorage().orElseThrow()));
    }

    @Test
    void resolvedLayoutsUseReferencedSpanAcrossAllRequiredGeometries() {
        assertCapacityBoundary(Shape.of(2, 3), LayoutDescriptor.contiguous(Shape.of(2, 3)), 6);
        assertCapacityBoundary(
                Shape.of(2, 3),
                LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 5, true),
                11);
        assertCapacityBoundary(
                Shape.of(2, 3),
                LayoutDescriptor.of(Shape.of(2, 3), new long[] {1, 2}, 0, true),
                6);
        assertCapacityBoundary(
                Shape.of(2, 3),
                LayoutDescriptor.of(Shape.of(2, 3), new long[] {0, 1}, 0, true),
                3);
        assertCapacityBoundary(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar()), 1);

        Shape empty = Shape.of(2, 0, 4);
        LayoutDescriptor emptyWithOffset =
                LayoutDescriptor.of(empty, new long[] {0, 4, 1}, 9, true);
        Tensor zeroCapacity = new Tensor(
                new TensorId(99),
                resolved(DataType.FLOAT32, empty, emptyWithOffset),
                Optional.empty(),
                Optional.empty(),
                Optional.of(storage(DataType.FLOAT32, 0)));
        assertEquals(0, zeroCapacity.hostStorage().orElseThrow().elementCapacity());
    }

    @Test
    void unresolvedStaticAndDynamicLayoutsDoNotInventCapacityRequirements() {
        Tensor staticTensor = new Tensor(
                new TensorId(1),
                unresolved(DataType.FLOAT32, Shape.of(100, 100)),
                Optional.empty(),
                Optional.empty(),
                Optional.of(storage(DataType.FLOAT32, 0)));
        Shape dynamicShape = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));
        Tensor dynamicTensor = new Tensor(
                new TensorId(2),
                unresolved(DataType.FLOAT32, dynamicShape),
                Optional.empty(),
                Optional.empty(),
                Optional.of(storage(DataType.FLOAT32, 0)));

        assertAll(
                () -> assertEquals(0, staticTensor.hostStorage().orElseThrow().elementCapacity()),
                () -> assertEquals(0, dynamicTensor.hostStorage().orElseThrow().elementCapacity()),
                () -> assertTrue(staticTensor.descriptor().layout().isEmpty()),
                () -> assertTrue(dynamicTensor.descriptor().layout().isEmpty()));
    }

    @Test
    void acceptsReadOnlyStorageWithoutWritingIt() {
        MemorySegment readOnly = MemorySegment.ofArray(new byte[4]).asReadOnly();
        HostTensorStorage storage = new MemorySegmentStorage(DataType.FLOAT32, 1, readOnly);
        Tensor tensor = new Tensor(
                new TensorId(1),
                resolved(DataType.FLOAT32, Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar())),
                Optional.empty(),
                Optional.empty(),
                Optional.of(storage));

        assertAll(
                () -> assertSame(storage, tensor.hostStorage().orElseThrow()),
                () -> assertTrue(tensor.hostStorage().orElseThrow().isReadOnly()));
    }

    @Test
    void observesLateStorageDeathAndCanClearWithoutOwningTheScope() {
        Arena arena = Arena.ofConfined();
        MemorySegment segment = arena.allocate(4, 1);
        HostTensorStorage storage = new MemorySegmentStorage(DataType.FLOAT32, 1, segment);
        Tensor tensor = new Tensor(
                new TensorId(1),
                resolved(DataType.FLOAT32, Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar())),
                Optional.empty(),
                Optional.empty(),
                Optional.of(storage));
        arena.close();

        assertAll(
                () -> assertSame(storage, tensor.hostStorage().orElseThrow()),
                () -> assertFalse(tensor.hostStorage().orElseThrow().isAlive()),
                () -> assertThrows(IllegalStateException.class, () -> segment.get(JAVA_BYTE, 0)));
        assertSame(storage, tensor.clearHostStorage().orElseThrow());
        assertTrue(tensor.hostStorage().isEmpty());
    }

    @Test
    void sharedStorageAliasesRemainIndependentTensorAssociations() {
        MemorySegment segment = MemorySegment.ofArray(new byte[] {1, 2});
        HostTensorStorage shared = new MemorySegmentStorage(DataType.BOOL, 2, segment);
        Tensor first = new Tensor(
                new TensorId(1), unresolved(DataType.BOOL, Shape.of(2)), Optional.empty(),
                Optional.empty(),
                Optional.of(shared));
        Tensor second = new Tensor(
                new TensorId(2), unresolved(DataType.BOOL, Shape.of(2)), Optional.empty(),
                Optional.empty(),
                Optional.of(shared));

        segment.set(JAVA_BYTE, 1, (byte) 9);
        first.clearHostStorage();

        assertAll(
                () -> assertTrue(first.hostStorage().isEmpty()),
                () -> assertSame(shared, second.hostStorage().orElseThrow()),
                () -> assertEquals(9, second.hostStorage().orElseThrow().segment().get(JAVA_BYTE, 1)));
    }

    @Test
    void usesObjectIdentityEvenWithEqualStableMetadata() throws NoSuchMethodException {
        TensorId firstId = new TensorId(3);
        TensorId equalId = new TensorId(3);
        TensorDescriptor descriptor = unresolved(DataType.FLOAT32, Shape.of(2));
        HostTensorStorage storage = storage(DataType.FLOAT32, 0);
        Tensor first = new Tensor(
                firstId, descriptor, Optional.of("x"), Optional.empty(), Optional.of(storage));
        Tensor second = new Tensor(
                equalId, descriptor, Optional.of("x"), Optional.empty(), Optional.of(storage));

        assertAll(
                () -> assertNotSame(firstId, equalId),
                () -> assertEquals(firstId, equalId),
                () -> assertEquals(first, first),
                () -> assertNotEquals(first, second),
                () -> assertEquals(System.identityHashCode(first), first.hashCode()),
                () -> assertEquals(Object.class,
                        Tensor.class.getMethod("equals", Object.class).getDeclaringClass()),
                () -> assertEquals(Object.class,
                        Tensor.class.getMethod("hashCode").getDeclaringClass()));
    }

    @Test
    void diagnosticTextIsStableMetadataOnlyAcrossStorageTransitionsAndDeath() {
        Arena arena = Arena.ofConfined();
        HostTensorStorage scoped = new MemorySegmentStorage(
                DataType.FLOAT32, 1, arena.allocate(4, 1));
        Tensor tensor = new Tensor(
                new TensorId(42),
                resolved(DataType.FLOAT32, Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar())),
                Optional.of("  result  "),
                Optional.empty(),
                Optional.empty());
        String absent = tensor.toString();
        tensor.replaceHostStorage(scoped);
        String present = tensor.toString();
        arena.close();
        String dead = tensor.toString();
        tensor.clearHostStorage();
        String cleared = tensor.toString();

        assertAll(
                () -> assertEquals(absent, present),
                () -> assertEquals(absent, dead),
                () -> assertEquals(absent, cleared),
                () -> assertTrue(absent.contains("Tensor[")),
                () -> assertTrue(absent.contains("id=TensorId[value=42]")),
                () -> assertTrue(absent.contains("descriptor=TensorDescriptor[")),
                () -> assertTrue(absent.contains("label=Optional[result]")),
                () -> assertFalse(absent.contains("hostStorage")),
                () -> assertFalse(absent.contains("MemorySegment")),
                () -> assertFalse(absent.contains("alive")),
                () -> assertFalse(absent.contains("graph")),
                () -> assertFalse(absent.contains("runtime")));
    }

    private static void assertCapacityBoundary(
            Shape shape, LayoutDescriptor layout, long requiredCapacity) {
        TensorDescriptor descriptor = resolved(DataType.FLOAT32, shape, layout);
        HostTensorStorage exact = storage(DataType.FLOAT32, requiredCapacity);
        Tensor accepted = new Tensor(
                new TensorId(requiredCapacity),
                descriptor,
                Optional.empty(),
                Optional.empty(),
                Optional.of(exact));
        assertSame(exact, accepted.hostStorage().orElseThrow());

        if (requiredCapacity > 0) {
            HostTensorStorage tooSmall = storage(DataType.FLOAT32, requiredCapacity - 1);
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new Tensor(
                            new TensorId(requiredCapacity + 100),
                            descriptor,
                            Optional.empty(),
                            Optional.empty(),
                            Optional.of(tooSmall)));
            assertEquals(
                    "hostStorage element capacity is smaller than resolved layout span: required="
                            + requiredCapacity
                            + ", actual="
                            + (requiredCapacity - 1),
                    failure.getMessage());
        }

        HostTensorStorage larger = storage(DataType.FLOAT32, requiredCapacity + 1);
        Tensor largerAccepted = new Tensor(
                new TensorId(requiredCapacity + 200),
                descriptor,
                Optional.empty(),
                Optional.empty(),
                Optional.of(larger));
        assertSame(larger, largerAccepted.hostStorage().orElseThrow());
    }

    private static TensorDescriptor unresolved(DataType dataType, Shape shape) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), false);
    }

    private static TensorDescriptor resolved(
            DataType dataType, Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(dataType, shape, Optional.of(layout), false);
    }

    private static HostTensorStorage storage(DataType dataType, long capacity) {
        long byteSize = Math.multiplyExact(capacity, dataType.byteWidth());
        MemorySegment segment = MemorySegment.ofArray(new byte[Math.toIntExact(byteSize)]);
        return new MemorySegmentStorage(dataType, capacity, segment);
    }

    private enum SampleKind implements OperationKind {
        SAMPLE
    }
}
