import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import Tensor.Tensor;

import static org.junit.jupiter.api.Assertions.*;

public class AllOpsTest {

    private Tensor a, b, c;

    private Tensor tensor(double[] data) {
        Tensor t = new Tensor(data, new int[]{data.length}, null, "t");
        t.setRequiresGrad(true);
        return t;
    }

    @BeforeEach
    public void setUp() {
        a = b = c = null;
    }

    @Test
    public void testAddForwardAndBackward() {
        a = tensor(new double[]{1.0, 2.0});
        b = tensor(new double[]{3.0, 4.0});
        c = a.add(b);
        c.compute();
        c.getCompiledGraph().setTrainingModeOff();
        c.compute();
        assertArrayEquals(new double[]{4.0, 6.0}, c.getData(), 1e-9);

        c.getCompiledGraph().setTrainingModeOn();
        c.compute();
        assertArrayEquals(new double[]{1.0, 1.0}, a.getGradient().getData(), 1e-9);
        assertArrayEquals(new double[]{1.0, 1.0}, b.getGradient().getData(), 1e-9);
    }

    @Test
    public void testSubForwardAndBackward() {
        a = tensor(new double[]{5.0, 7.0});
        b = tensor(new double[]{2.0, 3.0});
        c = a.sub(b);
        c.compute();
        c.getCompiledGraph().setTrainingModeOff();
        c.compute();
        assertArrayEquals(new double[]{3.0, 4.0}, c.getData(), 1e-9);

        c.getCompiledGraph().setTrainingModeOn();
        c.compute();
        assertArrayEquals(new double[]{1.0, 1.0}, a.getGradient().getData(), 1e-9);
        assertArrayEquals(new double[]{-1.0, -1.0}, b.getGradient().getData(), 1e-9);
    }

    @Test
    public void testMulForwardAndBackward() {
        a = tensor(new double[]{2.0, 3.0});
        b = tensor(new double[]{4.0, 5.0});
        c = a.mul(b);
        c.compute();
        c.getCompiledGraph().setTrainingModeOff();
        c.compute();
        assertArrayEquals(new double[]{8.0, 15.0}, c.getData(), 1e-9);

        c.getCompiledGraph().setTrainingModeOn();
        c.compute();
        assertArrayEquals(new double[]{4.0, 5.0}, a.getGradient().getData(), 1e-9);
        assertArrayEquals(new double[]{2.0, 3.0}, b.getGradient().getData(), 1e-9);
    }

    @Test
    public void testDivForwardAndBackward() {
        a = tensor(new double[]{8.0, 9.0});
        b = tensor(new double[]{2.0, 3.0});
        c = a.div(b);
        c.compute();
        c.getCompiledGraph().setTrainingModeOff();
        c.compute();
        assertArrayEquals(new double[]{4.0, 3.0}, c.getData(), 1e-9);

        c.getCompiledGraph().setTrainingModeOn();
        c.compute();
        assertArrayEquals(new double[]{0.5, 0.3333333333}, a.getGradient().getData(), 1e-9);
        assertArrayEquals(new double[]{-2.0, -1.0}, b.getGradient().getData(), 1e-9);
    }

    @Test
    public void testLogForwardAndBackward() {
        a = tensor(new double[]{Math.E, Math.E * Math.E});
        c = a.log();
        c.compute();
        c.getCompiledGraph().setTrainingModeOff();
        c.compute();
        assertArrayEquals(new double[]{1.0, 2.0}, c.getData(), 1e-9);

        c.getCompiledGraph().setTrainingModeOn();
        c.compute();
        assertArrayEquals(new double[]{1.0 / Math.E, 1.0 / (Math.E * Math.E)}, a.getGradient().getData(), 1e-9);
    }

    @Test
    public void testExpForwardAndBackward() {
        a = tensor(new double[]{0.0, 1.0});
        c = a.exp();
        c.compute();
        c.getCompiledGraph().setTrainingModeOff();
        c.compute();
        assertArrayEquals(new double[]{1.0, Math.exp(1.0)}, c.getData(), 1e-9);

        c.getCompiledGraph().setTrainingModeOn();
        c.compute();
        assertArrayEquals(new double[]{1.0, Math.exp(1.0)}, a.getGradient().getData(), 1e-9);
    }

    @Test
    public void testPowForwardAndBackward() {
        a = tensor(new double[]{2.0, 3.0});
        c = a.pow(3.0);
        c.compute();
        c.getCompiledGraph().setTrainingModeOff();
        c.compute();
        assertArrayEquals(new double[]{8.0, 27.0}, c.getData(), 1e-9);

        c.getCompiledGraph().setTrainingModeOn();
        c.compute();
        assertArrayEquals(new double[]{12.0, 27.0}, a.getGradient().getData(), 1e-9);
    }

    @Test
    public void testSumForward() {
        a = tensor(new double[]{1.0, 2.0, 3.0, 4.0});
        Tensor s = a.sum(0);
        s.compute();

        assertEquals(0, s.getShape().length);
        assertArrayEquals(new double[]{10.0}, s.getData(), 1e-9);
    }
}
