package Operations;

import Backend.ComputeBackend;
import Graph.FusedOperationGenerator;
import Tensor.Tensor;
import Utils.CustomClassLoader;

import java.io.File;
import java.io.FileOutputStream;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.*;

public class FusedOperation implements Operation {

    private final String expression;

    Method gradient;
    iFusedOperation instance=null;
    List<Tensor> cluster=null;
    private final double[][] inputsArr;

    public FusedOperation(List<Tensor> cluster, Tensor root) {
        this.expression = generateExpression(cluster);
        this.cluster=cluster;
        this.inputsArr = new double[findInputTensors(cluster).size()][];


        try {
            //byte[] byteCode = SimpleByteCodeGenerator.generateCustomClass(cluster);
            byte[] byteCode = FusedOperationGenerator.generate(cluster,root);
            String className = "Operations.fusedOperationClass";
            CustomClassLoader loader = new CustomClassLoader();
            Class<?> customClass = loader.define(className, byteCode);
            //forward = customClass.getMethod("apply", double[][].class);
            Constructor constructor = customClass.getConstructor(List.class);
            this.instance= (iFusedOperation) constructor.newInstance(cluster);
            new File("Operations").mkdirs();
            try (FileOutputStream fos = new FileOutputStream("Operations/fusedOperationClass.class")) {
                fos.write(byteCode);
            }
            catch (Exception e){
                e.printStackTrace();
            }


            //Method gradient = customClass.getMethod("gradient", double[][].class, double[].class);

        } catch (Throwable e) {
            throw new RuntimeException(e);
        }


    }


    //returns list of external inputs
    private List<Tensor> findInputTensors(List<Tensor> cluster) {
        List<Tensor> allPrevTensors = new ArrayList<>();
        for (Tensor tensor : cluster) {
            if (tensor.getPrevTensors() != null) {
                allPrevTensors.addAll(tensor.getPrevTensors());
                /*
                for (Tensor prevTensor : tensor.getPrevTensors()) {
                    allPrevTensors.add(prevTensor);
                }

                 */
            }
        }

        List<Tensor> inputTensors = new ArrayList<>();
        for (Tensor tensor : allPrevTensors) {
            if (!cluster.contains(tensor) && !inputTensors.contains(tensor)) {
                inputTensors.add(tensor);
            }
        }
        return inputTensors;
    }

    private String generateExpression(List<Tensor> cluster) {
        Map<Tensor, String> tensorNames = new HashMap<>();
        List<Tensor> inputTensors = findInputTensors(cluster);
        for (int i = 0; i < inputTensors.size(); i++) {
            tensorNames.put(inputTensors.get(i), "inputs[" + i + "][i]");
        }

        for (Tensor tensor : cluster) {
            Operation op = tensor.getOperation();
            List<Tensor> prevTensors = tensor.getPrevTensors();

            if (op instanceof add) {
                tensorNames.put(tensor, "(" + tensorNames.get(prevTensors.getFirst()) + " + " + tensorNames.get(prevTensors.getLast()) + ")");
            } else if (op instanceof sub) {
                tensorNames.put(tensor, "(" + tensorNames.get(prevTensors.getFirst()) + " - " + tensorNames.get(prevTensors.getLast()) + ")");
            } else if (op instanceof mul) {
                tensorNames.put(tensor, "(" + tensorNames.get(prevTensors.getFirst()) + " * " + tensorNames.get(prevTensors.getLast()) + ")");
            } else if (op instanceof div) {
                tensorNames.put(tensor, "(" + tensorNames.get(prevTensors.getFirst()) + " / " + tensorNames.get(prevTensors.getLast()) + ")");
            } else if (op instanceof log) {
                tensorNames.put(tensor, "Math.operations.log(" + tensorNames.get(prevTensors.getFirst()) + ")");
            } else if (op instanceof exp) {
                tensorNames.put(tensor, "Math.operations.exp(" + tensorNames.get(prevTensors.getFirst()) + ")");
            }
            else if (op instanceof pow) {
                tensorNames.put(tensor, "Math.operations.pow(" + tensorNames.get(prevTensors.getFirst()) + ","+((pow) op).exponent+")");
            }
        }

        // Poslední tensor v clusteru je výstupní tensor, jeho výraz je finální výpočet
        return tensorNames.get(cluster.get(cluster.size() - 1));
    }

    @Override
    public boolean isElementWise() {
        return true;
    }

    @Override
    public void apply(List<Tensor> inputs,Tensor node) {
        try {
            //intermitents = (Object[]) forward.invoke(customInstance, (Object) inputsArr);
            //instance.apply(inputs,node);
        }
        catch (Exception e) {
            throw new RuntimeException(e);
        }

    }


    @Override
    public void gradient(List<Tensor> inputs,Tensor node){
        int size=node.getPrevTensors().size();
        List<Tensor> prev=node.getPrevTensors();
        double [][] inputsArr = new double[size][];
        double [][] inputsGradients = new double[size][];
        double [] out=node.getGradient().getData();
        for (int i = 0; i < inputsArr.length; i++) {
            inputsArr[i] = prev.get(i).getData();
            inputsGradients[i] = prev.get(i).getGradient().getData();
        }

    }

    @Override
    public ComputeBackend getPreferredBackend() {
        return ComputeBackend.CPU;
    }

    @Override
    public boolean supportsBackend(ComputeBackend backend) {
        return backend == ComputeBackend.CPU;
    }

    @Override
    public String getExpression() {
        return expression;
    }

    @Override
    public boolean requiresOutputForGradient() {
        return true;
    }




}
