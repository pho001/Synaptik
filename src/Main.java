import Graph.FuseElementWise;
import Graph.GraphOptimizer;
import Graph.OptimizationRule;
import Graph.OptimizerFacory;
import Tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;



public class Main {
    public static void main(String[] args) {
        double[][] dA = {
                {1, 2, 3}, {1, 2, 3},
        };

        double [][] dB= {
                {4, 5, 6}, {4, 5, 6}
        };

        double [][] dC= {
                {7, 8, 9}, {7, 8, 9}
        };
        int repeats=1000;

        List <Tensor> inputs=new ArrayList<>();


        Tensor A=new Tensor(new int[]{10,10,10},null,"data A");
        Tensor B=new Tensor(new int[]{10,10,10},null,"data B");
        Tensor C=new Tensor(new int[]{10,10,10},null,"data C");
        //Tensor A=new Tensor(new double[]{5},null,"data A");
        //Tensor B=new Tensor(new double[]{2},null,"data B");
        //Tensor C=new Tensor(new double[]{2},null,"data C");



        A.setData(randomData(A.calculateSize(A.getShape())));
        B.setData(randomData(B.calculateSize(B.getShape())));
        C.setData(randomData(C.calculateSize(C.getShape())));
        A.setRequiresGrad(true);
        B.setRequiresGrad(true);
        C.setRequiresGrad(true);


        //A.setRequiresGrad(true);
        inputs.add(A);
        inputs.add(B);
        inputs.add(C);

        double[][] testInputs={A.getData(),B.getData(),C.getData()};


        //Tensor D=A.add(B).add(C).add(A).pow(0.5).sum();
        Tensor Ta1=A.div(B);
        Tensor Ta2=A.sub(C);
        Tensor Ta3=B.add(C);
        Tensor Ta4=Ta1.div(Ta2);

        Tensor Ta5=Ta3.mul(Ta4);

        Tensor Ta6=Ta4.add(Ta5);
        Tensor Ta7=Ta6.pow(2);
        //Tensor Ta8=Ta6.pow(2);


        double [] arr1 = randomData(1_000_000);
        double [] arr2 = randomData(1_000_000);
        double [] arr3 = randomData(1_000_000);
        double [] arr4 = randomData(1_000_000);
        double [] arr5 = randomData(1_000_000);
        double [] arr6 = randomData(1_000_000);
        double[][] a={arr1 ,arr2, arr3,arr4,arr5,arr6};








        long startTimeForward=0;
        long endTimeForward=0;

        long startTimeBackward=0;
        long endTimeBackward=0;
        double sumForward=0;
        double sumBackward=0;
        for (int k=0;k<repeats;k++) {
            startTimeForward = System.nanoTime();
            Ta7.compute();
            //calculate1(inputs, Ta7, a);
            //calculate3(inputs, Ta7);
            endTimeForward = System.nanoTime();
            startTimeBackward= System.nanoTime();
            Ta7.backward();
            endTimeBackward= System.nanoTime();

            double durationForward = (endTimeForward - startTimeForward) / 1_000_000.0;
            double durationBackward = (endTimeBackward - startTimeBackward) / 1_000_000.0;
            sumForward+=durationForward;
            sumBackward+=durationBackward;

        }
        double[] grad1=new double[A.getFlatDataSize()];
        double[] val1=new double[A.getGradient().getFlatDataSize()];

        System.arraycopy(A.getGradient().getData(), 0, grad1, 0, A.getFlatDataSize());
        System.arraycopy(Ta7.getData(), 0, val1, 0, Ta7.getFlatDataSize());

        System.out.println(
                "No Optimizer: Cluster executed in (average)\n" +
                        "Forward:   " + (sumForward / repeats) + " ms\n" +
                        "Backward:  " + (sumBackward / repeats) + " ms\n" +
                        "Both:  " + ((sumBackward+sumForward)/ repeats) + " ms\n"
        );


        GraphOptimizer optimizer=new GraphOptimizer();
        optimizer.addRule(OptimizerFacory.addFuseElementWise());
        startTimeForward=0;
        endTimeForward=0;

        startTimeBackward=0;
        endTimeBackward=0;
        sumForward=0;
        sumBackward=0;

        for (int k=0;k<repeats;k++) {
            startTimeForward = System.nanoTime();
            Ta7.compute(optimizer);
            endTimeForward = System.nanoTime();
            startTimeBackward= System.nanoTime();
            Ta7.backward();
            endTimeBackward= System.nanoTime();
            double durationForward = (endTimeForward - startTimeForward) / 1_000_000.0;
            double durationBackward = (endTimeBackward - startTimeBackward) / 1_000_000.0;
            sumForward+=durationForward;
            sumBackward+=durationBackward;
        }

        System.out.println(
                "Optimizer: Cluster executed in (average)\n" +
                        "Forward:   " + (sumForward / repeats) + " ms\n" +
                        "Backward:  " + (sumBackward / repeats) + " ms\n" +
                        "Both:  " + ((sumBackward+sumForward)/ repeats) + " ms\n"
        );
        double[] grad2=new double[A.getGradient().getFlatDataSize()];
        double[] val2=new double[Ta7.getFlatDataSize()];

        System.arraycopy(A.getGradient().getData(), 0, grad2, 0, A.getFlatDataSize());
        System.arraycopy(Ta7.getData(), 0, val2, 0, Ta7.getFlatDataSize());


        int i=0;
        int index=0;
        double sum=0;

        double resu=0;




        double zsum=0;
        double gradsum=0;

        for (i=0;i<Ta7.getFlatDataSize();i++) {
            zsum+=val1[i]-val2[i];
            gradsum+=grad1[i]-grad2[i];
        }






        System.out.println("Cluster executed in (average)" + sum/repeats + " ms");

        System.out.println("Result zero sum check:" +zsum);
        System.out.println("Grad zero sum check:" +gradsum);




        double [] pytelA=new double[]{10,20,30};
        double [] pytelB=new double[]{40,50,60};
        double [] pytelC=new double[]{70,80,90};

        double[][] inputy = {
                {1.0, 2.0},
                {3.0, 4.0},
                {5.0, 6.0}
        };




    }

    private static double[] randomData(int n){
        if (n <= 0) {
            throw new IllegalArgumentException("Délka pole musí být větší než 0.");
        }

        Random random = new Random();
        double[] randomData = new double[n];

        for (int i = 0; i < n; i++) {
            randomData[i] = (double) random.nextDouble(0,1); // Generuje náhodné číslo v rozsahu [1.0, 11.0)
            //randomData[i] = 1.0 + (5.0 - 1.0) * random.nextDouble();
        }

        return randomData;
    }

    public static void calculate1(List<Tensor> inputs, Tensor output,double [][] a){
        // Výstupní data
        double[] outData = output.getData();


        double[] arr1 = a[0];
        double[] arr2 = a[1];
        double[] arr3 = a[2];
        double[] arr4 = a[3];
        double[] arr5 = a[4];
        double[] arr6 = a[5];

        // Načtení vstupních polí z tensorů
        double[] input0 = (inputs.get(0)).getData();
        double[] input1 = (inputs.get(1)).getData();
        double[] input2 = (inputs.get(2)).getData();


        for (int i = 0; i < 1_000_000; i++) {
            arr1[i] = input2[i] / input0[i];
            arr2[i] = input2[i] * input1[i];
            arr3[i] = arr1[i] / arr2[i];
            arr4[i] = input0[i] * input1[i];
            arr5[i] = arr4[i] * arr3[i];
            arr6[i] = arr3[i] / arr5[i];
            outData[i] = arr6[i] * arr6[i];
        }
    }



    public static void calculate2(List<Tensor> inputs, Tensor output,double [][] a) {
        double[] outData = output.getData();

        double[] arr1 = a[0];
        double[] arr2 = a[1];
        double[] arr3 = a[2];
        double[] arr4 = a[3];
        double[] arr5 = a[4];
        double[] arr6 = a[5];


        double[] input0 = inputs.get(0).getData();
        double[] input1 = inputs.get(1).getData();
        double[] input2 = inputs.get(2).getData();

        // 1. arr1[i] = input2[i] / input0[i];
        for (int i = 0; i < 1_000_000; i++) {
            arr1[i] = input2[i] / input0[i];
        }

        // 2. arr2[i] = input2[i] * input1[i];
        for (int i = 0; i < 1_000_000; i++) {
            arr2[i] = input2[i] * input1[i];
        }

        // 3. arr3[i] = arr1[i] / arr2[i];
        for (int i = 0; i < 1_000_000; i++) {
            arr3[i] = arr1[i] / arr2[i];
        }

        // 4. arr4[i] = input0[i] * input1[i];
        for (int i = 0; i < 1_000_000; i++) {
            arr4[i] = input0[i] * input1[i];
        }

        // 5. arr5[i] = arr4[i] * arr3[i];
        for (int i = 0; i < 1_000_000; i++) {
            arr5[i] = arr4[i] * arr3[i];
        }

        // 6. arr6[i] = arr3[i] / arr5[i];
        for (int i = 0; i < 1_000_000; i++) {
            arr6[i] = arr3[i] / arr5[i];
        }

        // 7. outData[i] = arr6[i] * arr6[i];
        for (int i = 0; i < 1_000_000; i++) {
            outData[i] = arr6[i] * arr6[i];
        }
    }




    public static void calculate3(List<Tensor> inputs, Tensor output) {
        double[] outData = output.getData();

        // Získání nebo vytvoření pole intermediátů
        double[] intermediates = output.getIntermediates();
        if (intermediates == null) {
            intermediates = new double[6_000_000];
        }

        // Načtení vstupních polí z tensorů
        double[] input0 = inputs.get(0).getData();
        double[] input1 = inputs.get(1).getData();
        double[] input2 = inputs.get(2).getData();

        // Proměnné pro mezivýpočty
        double v10, v12, v14, v16, v18, v20;
        double i0;
        double i1;
        double i2;
        for (int i = 0; i < 1_000_000; i++) {

            v10 = input2[i] / input0[i];
            v12 = input2[i] * input1[i];
            v14 = v10 / v12;
            v16 = input0[i] * input1[i];
            v18 = v16 * v14;
            v20 = v14 / v18;

            outData[i] = v20 * v20;

            int base = i * 6;
            intermediates[base + 0] = v10;
            intermediates[base + 1] = v12;
            intermediates[base + 2] = v14;
            intermediates[base + 3] = v16;
            intermediates[base + 4] = v18;
            intermediates[base + 5] = v20;
        }

        output.setIntermediates(intermediates);

    }





}