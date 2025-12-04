package Graph;

import Tensor.Tensor;
import org.objectweb.asm.*;
import Operations.*;
import java.util.*;


public class SimpleByteCodeGenerator {


    public static byte[] generateCustomClass(List<Tensor> cluster) throws Throwable {

        List<Tensor> inputs=findInputTensors(cluster);

        ClassWriter cw = new ClassWriter(ClassWriter.COMPUTE_FRAMES | ClassWriter.COMPUTE_MAXS);
        cw.visit(Opcodes.V1_8, Opcodes.ACC_PUBLIC, "Operations/fusedOperationClass", null, "java/lang/Object", new String[] { "Operations/iFusedOperation" });

        // Constructor
        MethodVisitor mv = cw.visitMethod(Opcodes.ACC_PUBLIC, "<init>", "()V", null, null);
        mv.visitCode();
        mv.visitVarInsn(Opcodes.ALOAD, 0);
        mv.visitMethodInsn(Opcodes.INVOKESPECIAL, "java/lang/Object", "<init>", "()V", false);
        mv.visitInsn(Opcodes.RETURN);
        mv.visitMaxs(1, 1);
        mv.visitEnd();


        // Add method
        mv = cw.visitMethod(Opcodes.ACC_PUBLIC, "apply", "([[D)[Ljava/lang/Object;", null, null);

        //Byte code start
        mv.visitCode();
        boolean requiresGrad=false;
        for(Tensor t:cluster){
            if (t.getRequiresGrad())
                requiresGrad=true;
        }
        withIntermediates(mv,cluster,true);

        if (requiresGrad){
            withIntermediates(mv,cluster,true);
        }
        else{
            noIntermediates(mv,cluster);
        }




        mv.visitMaxs(10, 9);
        mv.visitEnd();
        //gradient(mv,cw,cluster,inputs);

        cw.visitEnd();
        byte[] classBytes = cw.toByteArray();
        //Class<?> dynamicClass = MethodHandles.lookup().defineClass(classBytes);

        return classBytes;


    }


    private static void withIntermediates(MethodVisitor mv,List<Tensor> cluster,boolean keepIntermittents) {

        List<Tensor> inputs=findInputTensors(cluster);


        /* Local variables Initialization */

        // Lokální proměnné:
        // 0 - this (implicitní pro instanční metodu)
        // 1 - input double[][] array
        // 2 - iCounter (int)
        // 3 - result array (double[])
        // 5 - tmparray - temp buffer

        // Získání délky input[0]

        int inputArr=1;
        int iCounter=2;
        int iOperations=3;
        int intermittentsArray=4;
        int tmpCounter=5;
        int finalResults = 6;
        int tmpVal=7;




        int iVals=inputs.getFirst().getFlatDataSize();                  //Values count
        int iOps= cluster.size();                                       //Cluster operations count
        int totalSize=iVals*iOps;



        //store operations count
        storeLocIntVar(mv,iOperations,iOps);


        // Create intermittent results array
        if (keepIntermittents)
            newArray(mv,intermittentsArray,totalSize,false);
        else
            newArray(mv,inputArr,iOps,false);

        // Create just results array
        newArray(mv, finalResults, iVals, false);

        // Loop init
        mv.visitInsn(Opcodes.ICONST_0);
        mv.visitVarInsn(Opcodes.ISTORE, iCounter);

        mv.visitInsn(Opcodes.ICONST_0);
        mv.visitVarInsn(Opcodes.ISTORE, tmpCounter);


        Label loopStart = new Label();
        Label loopEnd = new Label();

        // Loop start
        mv.visitLabel(loopStart);
        mv.visitVarInsn(Opcodes.ILOAD, iCounter);

        // Counter condition check
        if (totalSize <= 127) {
            mv.visitIntInsn(Opcodes.BIPUSH, totalSize);
        } else if (totalSize <= 32767) {
            mv.visitIntInsn(Opcodes.SIPUSH, totalSize);
        } else {
            mv.visitLdcInsn(totalSize);
        }

        mv.visitJumpInsn(Opcodes.IF_ICMPGE, loopEnd);






        // Input maps init
        Map<Tensor, Integer> inputIndices = new HashMap<>();    // For input tensors
        Map<Tensor, Integer> clusterIndices = new HashMap<>();  // For cluster tensors
        for (int i = 0; i < inputs.size(); i++) {
            inputIndices.put(inputs.get(i), i);
        }
        for (int i = 0; i < cluster.size(); i++) {
            clusterIndices.put(cluster.get(i), i);
        }




        for (int opIndex=0;opIndex<cluster.size();opIndex++) {

            Tensor tensor=cluster.get(opIndex);
            List<Tensor> prev= tensor.getPrevTensors();
            if (opIndex!=iOps-1) {
                mv.visitVarInsn(Opcodes.ALOAD, intermittentsArray);   //Stack: [resultsArray]
                mv.visitVarInsn(Opcodes.ILOAD, iCounter);       //Stack: [resultsArray, iCounter]
                if (opIndex != 0) {
                    mv.visitLdcInsn(opIndex);                       //Stack: [resultsArray, iCounter,opIndex]
                    mv.visitInsn(Opcodes.IADD);                     //Stack: [resultsArray, iCounter+opIndex]
                }
            }
            //loadArrayDouble(mv,resultsArray,iCounter,true);






            for (Tensor t : prev) {
                Integer index = inputIndices.get(t);
                if (index != null) {
                    // Load input tensor value
                    load2dArrayDouble(mv,inputArr,index,false,tmpCounter,true);   //Stack:[resultsArray, iCounter+opIndex,inputValue]
                }
                //pokud neobsahuje indexy na vnejsi tensory
                else {
                    int clusterIdx = clusterIndices.get(t);
                    mv.visitVarInsn(Opcodes.ALOAD, intermittentsArray); //Stack: [resultsArray, iCounter+opIndex,resultsArray]
                    mv.visitVarInsn(Opcodes.ILOAD, iCounter);     //Stack: [resultsArray, iCounter+opIndex,resultsArray,iCounter]
                    if (clusterIdx <= 127) {                    // [resultsArray, i, tmp, lastOpIdx]
                        mv.visitIntInsn(Opcodes.BIPUSH, clusterIdx);
                    } else if (clusterIdx <= 32767) {
                        mv.visitIntInsn(Opcodes.SIPUSH, clusterIdx);
                    } else {
                        mv.visitLdcInsn(clusterIdx);
                    }
                    //Stack: [resultsArray, iCounter+opIndex,resultsArray,iCounter,clusterIdx]

                    mv.visitInsn(Opcodes.IADD);                   //Stack: [resultsArray, iCounter+opIndex,resultsArray,iCounter+clusterIdx]
                    mv.visitInsn(Opcodes.DALOAD);                 //Stack: [resultsArray, iCounter+opIndex,prevValue]
                }
            }
            //Stack: [resultsArray, iCounter+opIndex,double1,double2] - pro binarni operace
            //nebo
            //Stack: [resultsArray, iCounter+opIndex,double] - pro unarni operace
            Operation operation=tensor.getOperation();


            switch (operation) {
                case add add -> mv.visitInsn(Opcodes.DADD);        // a[i] + b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,sum]
                case mul mul -> mv.visitInsn(Opcodes.DMUL);        // a[i] * b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,mul]
                case sub sub -> mv.visitInsn(Opcodes.DSUB);        // a[i] - b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,sub]
                case div div -> mv.visitInsn(Opcodes.DDIV);        // a[i] / b[i] (result is double) - Stack: [resultsArray, iCounter+opIndex,div]
                case log log -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math","log", "(D)D", false);
                case pow pow -> {
                    if (pow.getExponent()==0){
                        mv.visitInsn(Opcodes.POP);
                        mv.visitInsn(Opcodes.DCONST_1);
                    }
                    else if (pow.getExponent()==0.5 || pow.getExponent()==(double)1/2){
                        mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                    }
                    else if(pow.getExponent()==1){
                        continue;
                    }
                    else if (pow.getExponent()==2){
                        mv.visitInsn(Opcodes.DUP2);
                        mv.visitInsn(Opcodes.DMUL);
                    }
                    else {
                        mv.visitLdcInsn(pow.getExponent());
                        mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "pow", "(DD)D", false);
                    }
                }
                case exp exp -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "exp", "(D)D", false);
                /*
                case sqrt sqrt -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                case sin sin -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "sin", "(D)D", false);
                case cos cos -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "cos", "(D)D", false);
                case tan tan -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "tan", "(D)D", false);
                case asin asin -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "asin", "(D)D", false);
                case acos acos -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "acos", "(D)D", false);
                case atan atan -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "atan", "(D)D", false);
                case tanh tanh -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "tanh", "(D)D", false);

                 */

                default -> throw new UnsupportedOperationException();
            }

            //Stack: [resultsArray, iCounter+opIndex,opResultVal]

            if (opIndex == iOps - 1) {
                //Stack: [opResultVal]
                mv.visitVarInsn(Opcodes.DSTORE, tmpVal);
                // První zápis: resultsArray[iCounter + opIndex] = opResultVal
                mv.visitVarInsn(Opcodes.ALOAD, intermittentsArray);  // Stack: [resultsArray]
                mv.visitVarInsn(Opcodes.ILOAD, iCounter);      // Stack: [resultsArray, iCounter]
                if (opIndex != 0) {
                    mv.visitLdcInsn(opIndex);                       //Stack: [resultsArray, iCounter,opIndex]
                    mv.visitInsn(Opcodes.IADD);                     //Stack: [resultsArray, iCounter+opIndex]
                }
                mv.visitVarInsn(Opcodes.DLOAD, tmpVal);        // Stack: [resultsArray, iCounter+opIndex, opResultVal]
                mv.visitInsn(Opcodes.DASTORE);


                // Druhý zápis: finalResults[tmpCounter] = opResultVal
                mv.visitVarInsn(Opcodes.ALOAD, finalResults);  // Stack: [finalResults]
                mv.visitVarInsn(Opcodes.ILOAD, tmpCounter);    // Stack: [finalResults, tmpCounter]
                mv.visitVarInsn(Opcodes.DLOAD, tmpVal);        // Stack: [finalResults, tmpCounter, opResultVal]
                mv.visitInsn(Opcodes.DASTORE);                 // Stack: []
            }
            else {
                //Stack: [resultsArray, iCounter+opIndex,opResultVal]
                mv.visitInsn(Opcodes.DASTORE);
                //Stack: []
            }

        }




        // Inkrementace indexu - innkrement o iOps
        mv.visitIincInsn(tmpCounter, 1);
        mv.visitIincInsn(iCounter, iOps);
        mv.visitJumpInsn(Opcodes.GOTO, loopStart);

        mv.visitLabel(loopEnd);
        /*
        mv.visitVarInsn(Opcodes.ALOAD, resultsArray);
        mv.visitInsn(Opcodes.ARETURN);

         */
        // Vytvoření Object[] přímo (bez zbytečného obalení do Object)
        mv.visitInsn(Opcodes.ICONST_2);
        mv.visitTypeInsn(Opcodes.ANEWARRAY, "java/lang/Object");

        // Uložení prvního prvku (intermediates)
        mv.visitInsn(Opcodes.DUP);
        mv.visitInsn(Opcodes.ICONST_0);
        mv.visitVarInsn(Opcodes.ALOAD, finalResults);
        mv.visitInsn(Opcodes.AASTORE);

        // Uložení druhého prvku (finální výsledky)
        mv.visitInsn(Opcodes.DUP);
        mv.visitInsn(Opcodes.ICONST_1);
        mv.visitVarInsn(Opcodes.ALOAD, intermittentsArray);
        mv.visitInsn(Opcodes.AASTORE);

        // Návratová hodnota je již na zásobníku
        mv.visitInsn(Opcodes.ARETURN);


    }



    private static void noIntermediates(MethodVisitor mv,List<Tensor> cluster) {

        List<Tensor> inputs=findInputTensors(cluster);


        /* Local variables Initialization */

        // Lokální proměnné:
        // 0 - this (implicitní pro instanční metodu)
        // 1 - input double[][] array
        // 2 - iCounter (int)
        // 3 - result array (double[])
        // 5 - tmparray - temp buffer

        // Získání délky input[0]

        int inputArr=1;
        int iCounter=2;
        int iOperations=3;
        int resultsArray=4;
        int tmpArray=5;



        int iVals=inputs.getFirst().getFlatDataSize();                  //Values count
        int iOps= cluster.size();                                       //Cluster operations count
        int totalSize=iVals*iOps;





        mv.visitVarInsn(Opcodes.BIPUSH,iOps);                           //[Stack: iOps]
        mv.visitVarInsn(Opcodes.ISTORE,iOperations);                    //[Stack: ]


        // Create result array
        if (totalSize <= 127) {
            mv.visitIntInsn(Opcodes.BIPUSH, totalSize);
        } else if (totalSize <= 32767) {
            mv.visitIntInsn(Opcodes.SIPUSH, totalSize);
        } else {
            mv.visitLdcInsn(totalSize);
        }
        //[Stack: totalSize]
        mv.visitIntInsn(Opcodes.NEWARRAY, Opcodes.T_DOUBLE);
        mv.visitVarInsn(Opcodes.ASTORE, resultsArray);                  //[Stack: ]


        // buffer unit init
        if (iOps <= 127) {
            mv.visitIntInsn(Opcodes.BIPUSH, iOps);
        } else if (iOps <= 32767) {
            mv.visitIntInsn(Opcodes.SIPUSH, iOps);
        } else {
            mv.visitLdcInsn(iOps);
        }
        //[Stack: iOps]
        mv.visitIntInsn(Opcodes.NEWARRAY, Opcodes.T_DOUBLE);
        mv.visitVarInsn(Opcodes.ASTORE, tmpArray);


        // Loop init
        mv.visitInsn(Opcodes.ICONST_0);
        mv.visitVarInsn(Opcodes.ISTORE, iCounter);


        Label loopStart = new Label();
        Label loopEnd = new Label();

        // Loop start
        mv.visitLabel(loopStart);
        mv.visitVarInsn(Opcodes.ILOAD, iCounter);

        // Counter condition check
        if (iVals <= 127) {
            mv.visitIntInsn(Opcodes.BIPUSH, iVals);
        } else if (iVals <= 32767) {
            mv.visitIntInsn(Opcodes.SIPUSH, iVals);
        } else {
            mv.visitLdcInsn(iVals);
        }

        mv.visitJumpInsn(Opcodes.IF_ICMPGE, loopEnd);






        // Input maps init
        Map<Tensor, Integer> inputIndices = new HashMap<>();    // For input tensors
        Map<Tensor, Integer> clusterIndices = new HashMap<>();  // For cluster tensors
        for (int i = 0; i < inputs.size(); i++) {
            inputIndices.put(inputs.get(i), i);
        }
        for (int i = 0; i < cluster.size(); i++) {
            clusterIndices.put(cluster.get(i), i);
        }




        for (int opIndex=0;opIndex<cluster.size();opIndex++) {

            Tensor tensor=cluster.get(opIndex);
            List<Tensor> prev= tensor.getPrevTensors();
            mv.visitVarInsn(Opcodes.ALOAD, resultsArray);   //Stack: [resultsArray]
            mv.visitVarInsn(Opcodes.ILOAD, iCounter);       //Stack: [resultsArray, iCounter]
            mv.visitLdcInsn(opIndex);                       //Stack: [resultsArray, iCounter,opIndex]
            mv.visitInsn(Opcodes.IADD);                     //Stack: [resultsArray, iCounter+opIndex]

            //prepare buffer for future data storing
            mv.visitVarInsn(Opcodes.ALOAD, tmpArray);        //Stack: [resultsArray, iCounter+opIndex, tmpArray]

            if (opIndex <= 127) {
                mv.visitIntInsn(Opcodes.BIPUSH, opIndex);
            } else if (iOps <= 32767) {
                mv.visitIntInsn(Opcodes.SIPUSH, opIndex);
            } else {
                mv.visitLdcInsn(opIndex);
            }
            //  Stack: [resultsArray, iCounter+opIndex,tmpArray,opIndex]


            for (Tensor t : prev) {
                Integer index = inputIndices.get(t);
                if (index != null) {
                    // Load input tensor
                    mv.visitVarInsn(Opcodes.ALOAD, inputArr);       //Stack:[resultsArray, iCounter+opIndex,tmpArray,opIndex,inputArr]
                    mv.visitIntInsn(Opcodes.BIPUSH, index);         //Stack:[resultsArray, iCounter+opIndex,tmpArray,opIndex,inputArr, index]
                    mv.visitInsn(Opcodes.AALOAD);                   //Stack:[resultsArray, iCounter+opIndex,tmpArray,opIndex,inputArr[inputIndex]
                    mv.visitVarInsn(Opcodes.ILOAD, iCounter);       //Stack:[resultsArray, iCounter+opIndex,tmpArray,opIndex,inputArr[inputIndex][iCounter]]
                    mv.visitInsn(Opcodes.DALOAD);                   //Stack:[resultsArray, iCounter+opIndex,tmpArray,opIndex,inputValue]
                }
                //pokud neobsahuje indexy na vnejsi tensory
                else {
                    int clusterIdx = clusterIndices.get(t);
                    mv.visitVarInsn(Opcodes.ALOAD, tmpArray);     //Stack: [resultsArray, iCounter+opIndex,tmpArray,opIndex,tmpArray]
                    mv.visitIntInsn(Opcodes.BIPUSH, clusterIdx);  //Stack: [resultsArray, iCounter+opIndex,tmpArray,opIndex,tmpArray,clusterIndex]
                    mv.visitInsn(Opcodes.DALOAD);                 //Stack: [resultsArray, iCounter+opIndex,tmpArray,opIndex,tmpValue]
                }
            }
            //Stack: [resultsArray, iCounter+opIndex,tmpArray,opIndex,double1,double2] - pro binarni operace
            //nebo
            //Stack: [resultsArray, iCounter+opIndex,tmpArray,opIndex,double] - pro unarni operace
            Operation operation=tensor.getOperation();


            switch (operation) {
                case add add -> mv.visitInsn(Opcodes.DADD);        // a[i] + b[i] (result is double) - Stack: [tmpArray,opIndex,sum]
                case mul mul -> mv.visitInsn(Opcodes.DMUL);        // a[i] * b[i] (result is double) - Stack: [tmpArray,opIndex,mul]
                case sub sub -> mv.visitInsn(Opcodes.DSUB);        // a[i] - b[i] (result is double) - Stack: [tmpArray,opIndex,sub]
                case div div -> mv.visitInsn(Opcodes.DDIV);        // a[i] / b[i] (result is double) - Stack: [tmpArray,opIndex,div]
                case log log -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math","log", "(D)D", false);
                case pow pow -> {
                    if (pow.getExponent()==0){
                        mv.visitInsn(Opcodes.POP);
                        mv.visitInsn(Opcodes.DCONST_1);
                    }
                    else if (pow.getExponent()==0.5 || pow.getExponent()==(double)1/2){
                        mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                    }
                    else if(pow.getExponent()==1){
                        continue;
                    }
                    else if (pow.getExponent()==2){
                        mv.visitInsn(Opcodes.DUP2);
                        mv.visitInsn(Opcodes.DMUL);
                    }
                    else {
                        mv.visitLdcInsn(pow.getExponent());
                        mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "pow", "(DD)D", false);
                    }
                }
                case exp exp -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "exp", "(D)D", false);
                /*
                case sqrt sqrt -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "sqrt", "(D)D", false);
                case sin sin -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "sin", "(D)D", false);
                case cos cos -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "cos", "(D)D", false);
                case tan tan -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "tan", "(D)D", false);
                case asin asin -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "asin", "(D)D", false);
                case acos acos -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "acos", "(D)D", false);
                case atan atan -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "atan", "(D)D", false);
                case tanh tanh -> mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "tanh", "(D)D", false);

                 */

                default -> throw new UnsupportedOperationException();
            }

            //Stack: [resultsArray, iCounter+opIndex,tmpArray,opIndex,opResultVal (sum/mul/sub/...)]
            // Výstup ulož do tmp bufferu a nasledne do results na spravny index
            mv.visitInsn(Opcodes.DUP2_X2);     //Stack: [resultsArray, iCounter+opIndex,opResultVal,tmpArray,opIndex,opResultVal]
            mv.visitInsn(Opcodes.DASTORE);     //Stack: [resultsArray, iCounter+opIndex,opResultVal]
            mv.visitInsn(Opcodes.DASTORE);     //Stack: []


            //Stack: []
        }

        /*
        mv.visitVarInsn(Opcodes.ALOAD,resultsArray);
        mv.visitVarInsn(Opcodes.ILOAD, iCounter);
        //Stack: [results,iCounter]

        mv.visitVarInsn(Opcodes.ALOAD, tmpArray);       // [resultsArray, i, tmpArray]
        if (cluster.size() <= 127) {                    // [resultsArray, i, tmp, lastOpIdx]
            mv.visitIntInsn(Opcodes.BIPUSH, cluster.size() - 1);
        } else if (cluster.size() <= 32767) {
            mv.visitIntInsn(Opcodes.SIPUSH, cluster.size() - 1);
        } else {
            mv.visitLdcInsn(cluster.size() - 1);
        }
        mv.visitInsn(Opcodes.DALOAD);                   // [resultsArray, i, value]
        mv.visitInsn(Opcodes.DASTORE);
        */


        // Inkrementace indexu
        mv.visitIincInsn(iCounter, 1);
        mv.visitJumpInsn(Opcodes.GOTO, loopStart);

        mv.visitLabel(loopEnd);
        mv.visitVarInsn(Opcodes.ALOAD, resultsArray);
        mv.visitInsn(Opcodes.ARETURN);

    }



    private static void gradient(MethodVisitor mv,ClassWriter cw,List<Tensor> cluster,List <Tensor> inputs){

        mv = cw.visitMethod(Opcodes.ACC_PUBLIC, "gradient", "([[D[[D)V", null, null);
        mv.visitCode();

        /*
            Local variables:

         1 - inputs - double[][] array
         2 - gradients - double[][] array
         3 - result array (double[])
         4 - iCounter (int)
         5 - tmparray - temp buffer

        */


        int inputArr=1;
        int gradients=2;
        int values=3;
        int inputCols=4;
        int inputRows=5;
        int iCounter=6;




        // Počet řádků = inputArr.length
        mv.visitVarInsn(Opcodes.ALOAD, inputArr);     // načti double[][]
        mv.visitInsn(Opcodes.ARRAYLENGTH);            // délka pole (počet řádků)
        mv.visitVarInsn(Opcodes.ISTORE, inputRows);   // ulož do inputRows

        // Počet sloupců = inputArr[0].length
        mv.visitVarInsn(Opcodes.ALOAD, inputArr);     // načti double[][]
        mv.visitInsn(Opcodes.ICONST_0);               // index 0
        mv.visitInsn(Opcodes.AALOAD);                 // inputArr[0] → double[]
        mv.visitInsn(Opcodes.ARRAYLENGTH);            // délka pole (počet sloupců)
        mv.visitVarInsn(Opcodes.ISTORE, inputCols);   // ulož do inputCols


        // Inicializace indexu i=0
        mv.visitInsn(Opcodes.ICONST_0);
        mv.visitVarInsn(Opcodes.ISTORE, iCounter);

        Label loopStart = new Label();
        Label loopEnd = new Label();

        // Start smyčky
        mv.visitLabel(loopStart);
        mv.visitVarInsn(Opcodes.ILOAD, iCounter);
        mv.visitVarInsn(Opcodes.ILOAD, inputCols);
        mv.visitJumpInsn(Opcodes.IF_ICMPGE, loopEnd);



        Map<Tensor, Integer> inputsIndexMap = new HashMap<>();

        for (int i = 0; i < inputs.size(); i++) {
            inputsIndexMap.put(inputs.get(i), i);
        }

        for (int row = cluster.size() - 1; row >= 0; row--) {
            Tensor tensor = cluster.get(row);
            List<Tensor> prev = tensor.getPrevTensors();
            List<Integer> indexes = new ArrayList<>();
            //pro kazdy tensor nacti jeho predky
            for (Tensor t : prev) {
                Integer index = inputsIndexMap.get(t);
                if (index != null) {
                    indexes.add(index);
                }
            }

            //nacti operaci na danem tensoru
            Operation operation = tensor.getOperation();



            switch (operation) {
                case Operations.add add -> {
                    load2dArrayDouble(mv, gradients, indexes.get(0),false, iCounter,true);//Stack: [gradient[index][i]]   - sem budeme ukladat gradient prvniho operatoru

                    //derivace podle prvniho operatoru

                    load2dArrayDouble(mv, gradients, indexes.get(0),false, iCounter,true);    // gradient prvniho operatou
                    load2dArrayDouble(mv, gradients, row,true, iCounter,true);     //Stack: [gradient[index][i],gradient[index][i],gradient[row][i]]  - nacti gradient pro aktualni tensor
                    mv.visitInsn(Opcodes.DADD);
                    mv.visitInsn(Opcodes.DASTORE);

                    load2dArrayDouble(mv, gradients, indexes.get(1),false, iCounter,true);//Stack: [gradient[index][i]]   - sem budeme ukladat gradient druheho operatoru
                    //derivace podle prvniho operatoru
                    load2dArrayDouble(mv, gradients, indexes.get(1),false, iCounter,true);
                    load2dArrayDouble(mv, gradients, row,false, iCounter,true);     //Stack: [gradient[index][i],gradient[index][i],gradient[row][i]]
                    mv.visitInsn(Opcodes.DADD);
                    mv.visitInsn(Opcodes.DASTORE);

                }
                case mul mul -> {
                    // z=x*y
                    // ∂L/∂x += ∂L/∂z * y

                    load2dArrayDouble(mv, gradients, indexes.get(0),false, iCounter,true);        // CÍL pro DASTORE
                    load2dArrayDouble(mv, gradients, row,false, iCounter,true);                   // ∂L/∂x (aktuální gradient)
                    load2dArrayDouble(mv, gradients, indexes.get(0),false, iCounter,true);        // ∂L/∂z (gradient prvniho operandu)
                    load2dArrayDouble(mv, values, row,false, iCounter,true);                      // y
                    mv.visitInsn(Opcodes.DMUL);                                         // ∂L/∂z * y
                    mv.visitInsn(Opcodes.DADD);                                         // ∂L/∂x += ∂L/∂z * y
                    mv.visitInsn(Opcodes.DASTORE);                                      // uložíme do ∂L/∂x
                    // z=x*y
                    // ∂L/∂y += ∂L/∂z * x
                    load2dArrayDouble(mv, gradients, indexes.get(1),false, iCounter,true);        // CÍL pro DASTORE
                    load2dArrayDouble(mv, gradients, row,false, iCounter,true);                   // ∂L/∂y (aktuální gradient)
                    load2dArrayDouble(mv, gradients, indexes.get(1),false, iCounter,true);        // ∂L/∂z
                    load2dArrayDouble(mv, values, indexes.get(1),false, iCounter,true);           // x
                    mv.visitInsn(Opcodes.DMUL);                                         // ∂L/∂z * x
                    mv.visitInsn(Opcodes.DADD);                                         // ∂L/∂y += ∂L/∂z * x
                    mv.visitInsn(Opcodes.DASTORE);                                      // uložíme do ∂L/∂y
                }
                case sub sub ->{
                    // ∂L/∂x += ∂L/∂z * y

                    load2dArrayDouble(mv, gradients, indexes.get(0),false, iCounter,true);//Stack: [gradient[index][i]]   - sem budeme ukladat gradient prvniho operatoru

                    //derivace podle prvniho operatoru

                    load2dArrayDouble(mv, gradients, indexes.get(0),false, iCounter,true);
                    load2dArrayDouble(mv, gradients, row,false, iCounter,true);     //Stack: [gradient[index][i],gradient[index][i],gradient[row][i]]  - nacti gradient pro aktualni tensor
                    mv.visitInsn(Opcodes.DADD);
                    mv.visitInsn(Opcodes.DASTORE);

                    load2dArrayDouble(mv, gradients, indexes.get(1),false, iCounter,true);//Stack: [gradient[index][i]]   - sem budeme ukladat gradient prvniho operatoru
                    //derivace podle prvniho operatoru
                    load2dArrayDouble(mv, gradients, indexes.get(1),false, iCounter,true);
                    load2dArrayDouble(mv, gradients, row,false, iCounter,true);     //Stack: [gradient[index][i],gradient[index][i],gradient[row][i]]
                    mv.visitInsn(Opcodes.DNEG);
                    mv.visitInsn(Opcodes.DADD);
                    mv.visitInsn(Opcodes.DASTORE);                             // uložíme do ∂L/∂y
                }

                case Operations.div div ->{
                    // ∂L/∂x = ∂L/∂z * (1 / y)
                    load2dArrayDouble(mv, gradients, indexes.get(0),false, iCounter,true);   // Stack: [gradient[index0][i]]

                    // Derivace podle prvního operandu (x)
                    load2dArrayDouble(mv, gradients, row,false, iCounter,true);               // Stack: [gradient[index0][i], ∂L/∂z]
                    load2dArrayDouble(mv, values, indexes.get(1),false, iCounter,true);       // Stack: [gradient[index0][i], ∂L/∂z, y]
                    mv.visitInsn(Opcodes.DDIV);                                     // Stack: [gradient[index0][i], ∂L/∂z / y]
                    mv.visitInsn(Opcodes.DADD);                                     // Stack: [gradient[index0][i] + ∂L/∂z / y]
                    mv.visitInsn(Opcodes.DASTORE);                                   // Uložíme zpět do gradients[index0][i]

                    // ∂L/∂y = -∂L/∂z * (x / y^2)
                    load2dArrayDouble(mv, gradients, indexes.get(1),false, iCounter,true);    // Stack: [gradient[index1][i]]

                    // Derivace podle druhého operandu (y)
                    load2dArrayDouble(mv, gradients, row,false, iCounter,true);              // Stack: [gradient[index1][i], ∂L/∂z]
                    load2dArrayDouble(mv, values, indexes.get(0),false, iCounter,true);      // Stack: [gradient[index1][i], ∂L/∂z, x]
                    load2dArrayDouble(mv, values, indexes.get(1),false, iCounter,true);      // Stack: [gradient[index1][i], ∂L/∂z, x, y]
                    mv.visitInsn(Opcodes.DMUL);                                     // Stack: [gradient[index1][i], ∂L/∂z, x * y]
                    mv.visitInsn(Opcodes.DDIV);                                     // Stack: [gradient[index1][i], ∂L/∂z, x / y]
                    mv.visitInsn(Opcodes.DDIV);                                     // Stack: [gradient[index1][i], ∂L/∂z, x / y^2]
                    mv.visitInsn(Opcodes.DNEG);                                     // Stack: [gradient[index1][i], -∂L/∂z * (x / y^2)]
                    mv.visitInsn(Opcodes.DADD);                                     // Stack: [gradient[index1][i] - ∂L/∂z * (x / y^2)]
                    mv.visitInsn(Opcodes.DASTORE);                                   // Uložíme zpět do gradients[index1][i]
                }

                case log log -> {
                    // ∂L/∂x = ∂L/∂z * (1 / x)
                    load2dArrayDouble(mv, gradients, indexes.get(0),false, iCounter,true);   // Stack: [gradient[index0][i]]

                    // Derivace podle prvního operandu (x)
                    load2dArrayDouble(mv, gradients, row,false, iCounter,true);              // Stack: [gradient[index0][i], ∂L/∂z]
                    load2dArrayDouble(mv, values, indexes.get(0),false, iCounter,true);      // Stack: [gradient[index0][i], ∂L/∂z, x]
                    mv.visitInsn(Opcodes.DDIV);                                    // Stack: [gradient[index0][i], ∂L/∂z / x]
                    mv.visitInsn(Opcodes.DADD);                                    // Stack: [gradient[index0][i] + ∂L/∂z / x]
                    mv.visitInsn(Opcodes.DASTORE);                                 // Uložíme zpět do gradients[index0][i]
                }

                case exp exp ->{
                    // ∂L/∂x = ∂L/∂z * operations.exp(x)
                    load2dArrayDouble(mv, gradients, indexes.get(0),false, iCounter,true);   // Stack: [gradient[index0][i]]

                    // Derivace podle prvního operandu (x)
                    load2dArrayDouble(mv, gradients, row,false, iCounter,true);              // Stack: [gradient[index0][i], ∂L/∂z]
                    load2dArrayDouble(mv, values, indexes.get(0),false, iCounter,true);      // Stack: [gradient[index0][i], ∂L/∂z, x]
                    mv.visitInsn(Opcodes.DUP2);                                     // Stack: [gradient[index0][i], ∂L/∂z, x, x]
                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "operations.exp", "(D)D", false); // Stack: [gradient[index0][i], ∂L/∂z, x, operations.exp(x)]
                    mv.visitInsn(Opcodes.DMUL);                                     // Stack: [gradient[index0][i], ∂L/∂z * operations.exp(x)]
                    mv.visitInsn(Opcodes.DADD);                                     // Stack: [gradient[index0][i] + ∂L/∂z * operations.exp(x)]
                    mv.visitInsn(Opcodes.DASTORE);                                  // Uložíme zpět do gradients[index0][i]
                }

                case pow pow -> {
                    // Derivace podle x: ∂L/∂x = ∂L/∂z * y * x^(y-1)

                    load2dArrayDouble(mv, gradients, indexes.get(0),false, iCounter,true);  // Stack: [gradient[index0][i]]

                    // Derivace podle x (z = x^y): ∂L/∂x = ∂L/∂z * y * x^(y-1)
                    load2dArrayDouble(mv, gradients, row,false, iCounter,true);             // Stack: [gradient[index0][i], ∂L/∂z]
                    load2dArrayDouble(mv, values, indexes.get(0),false, iCounter,true);     // Stack: [gradient[index0][i], ∂L/∂z, x]
                    mv.visitLdcInsn(pow.getExponent());                                // Stack: [gradient[index0][i], ∂L/∂z, x,operations.pow]
                    mv.visitInsn(Opcodes.DUP2);                                    // Stack: [gradient[index0][i], ∂L/∂z, x, y, y]
                    mv.visitInsn(Opcodes.DCONST_1);                                 // Stack: [gradient[index0][i], ∂L/∂z, x, y, y,1]
                    mv.visitInsn(Opcodes.DSUB);                                    // Stack: [gradient[index0][i], ∂L/∂z, x, y, y-1]
                    mv.visitMethodInsn(Opcodes.INVOKESTATIC, "java/lang/Math", "operations.pow", "(DD)D", false); // Stack: [gradient[index0][i], ∂L/∂z, x, y, x^(y-1)]
                    mv.visitInsn(Opcodes.DMUL);                                    // Stack: [gradient[index0][i], ∂L/∂z, x, y * x^(y-1)]
                    mv.visitInsn(Opcodes.DMUL);                                    // Stack: [gradient[index0][i], ∂L/∂z * y * x^(y-1)]
                    mv.visitInsn(Opcodes.DADD);                                    // Stack: [gradient[index0][i] + ∂L/∂z * y * x^(y-1)]
                    mv.visitInsn(Opcodes.DASTORE);                                 // Uložíme do gradients[index0][i]
                }
                default -> throw new UnsupportedOperationException();


            }


        }
        mv.visitIincInsn(iCounter, 1);
        mv.visitJumpInsn(Opcodes.GOTO, loopStart);

        mv.visitLabel(loopEnd);
        mv.visitVarInsn(Opcodes.ALOAD, 3);
        mv.visitInsn(Opcodes.ARETURN);

        mv.visitMaxs(6, 5);
        mv.visitEnd();

    }

    //returns list of external inputs
    private static List<Tensor> findInputTensors(List<Tensor> cluster) {

        Set<Tensor> allPrevTensors = new HashSet<>();
        for (Tensor tensor : cluster) {
            if (tensor.getPrevTensors() != null) {
                allPrevTensors.addAll(tensor.getPrevTensors());
            }
        }

        List<Tensor> inputTensors = new ArrayList<>();
        for (Tensor tensor : allPrevTensors) {
            if (!cluster.contains(tensor)) {
                inputTensors.add(tensor);
            }
        }

        return inputTensors;
    }




    // load element in double[][] array
    private static void load2dArrayDouble(MethodVisitor mv, int localVarPosition, int rowIndex, boolean isRowLocalVar, int colIndex, boolean isColLocalVar) {
        mv.visitVarInsn(Opcodes.ALOAD, localVarPosition); //Stack:[...,array]
        if (isRowLocalVar) {
            mv.visitVarInsn(Opcodes.ILOAD, rowIndex);
        }
        else {
            if (rowIndex >= -128 && rowIndex <= 127) {
                mv.visitIntInsn(Opcodes.BIPUSH, rowIndex);
            } else if (rowIndex>=-32768 && rowIndex <= 32767) {
                mv.visitIntInsn(Opcodes.SIPUSH, rowIndex);
            } else {
                mv.visitLdcInsn(rowIndex);
            }
        }
        //Stack:[...,array,rowIndex]

        mv.visitInsn(Opcodes.AALOAD);       //Stack:[...,array[rowindex]]
        if (isColLocalVar) {
            mv.visitVarInsn(Opcodes.ILOAD, colIndex);
        }
        else {
            if (colIndex >= -128 && colIndex <= 127) {
                mv.visitIntInsn(Opcodes.BIPUSH, colIndex);
            } else if (colIndex>=-32768 && colIndex <= 32767) {
                mv.visitIntInsn(Opcodes.SIPUSH, colIndex);
            } else {
                mv.visitLdcInsn(colIndex);
            }
        }
        //Stack:[...,array[rowindex],colIndex]
        mv.visitInsn(Opcodes.DALOAD);   //Stack:[...,double]
    }

    private static void loadArrayDouble(MethodVisitor mv, int localVarPosition,  int col, boolean isLocalVar) {
        mv.visitVarInsn(Opcodes.ALOAD, localVarPosition);     //Stack:[...,array]
        if (isLocalVar) {
            mv.visitVarInsn(Opcodes.ILOAD, col);
        }
        else {
            if (col >= -128 && col <= 127) {
                mv.visitIntInsn(Opcodes.BIPUSH, col);
            } else if (col>=-32768 && col <= 32767) {
                mv.visitIntInsn(Opcodes.SIPUSH, col);
            } else {
                mv.visitLdcInsn(col);
            }
        }
        //Stack:[...,array,colIndex]
        mv.visitInsn(Opcodes.DALOAD);   //Stack:[...,double]
    }

    private static void newArray(MethodVisitor mv, int localVarPosition, int cols, boolean isLocalVariable){
        if (isLocalVariable) {
            mv.visitVarInsn(Opcodes.ILOAD, cols);
        }
        else{
            if (cols >= -128 && cols <= 127) {
                mv.visitIntInsn(Opcodes.BIPUSH, cols);
            } else if (cols>=-32768 && cols <= 32767) {
                mv.visitIntInsn(Opcodes.SIPUSH, cols);
            } else {
                mv.visitLdcInsn(cols);
            }
        }

        //[Stack: cols]
        mv.visitIntInsn(Opcodes.NEWARRAY, Opcodes.T_DOUBLE);
        mv.visitVarInsn(Opcodes.ASTORE, localVarPosition);
    }




    private static void storeLocIntVar(MethodVisitor mv, int locVarPos,int variable){
        if (variable >= -128 && variable <= 127) {
            mv.visitIntInsn(Opcodes.BIPUSH, variable);
        } else if (variable >= -32768 && variable <= 32767) {
            mv.visitIntInsn(Opcodes.SIPUSH, variable);
        } else {
            mv.visitLdcInsn(variable);
        }
                          //[Stack: iOps]
        mv.visitVarInsn(Opcodes.ISTORE,locVarPos);
    }


    /*
    private static void storeArrayElement(MethodVisitor mv, int localVarPosition, int colIndex, boolean isLocalVar,double value) {

        mv.visitVarInsn(Opcodes.ALOAD, localVarPosition);     //Stack:[...,array]
        if (isLocalVar) {
            mv.visitVarInsn(Opcodes.ILOAD, colIndex);
        }
        else {
            if (colIndex <= 127) {
                mv.visitIntInsn(Opcodes.BIPUSH, colIndex);
            } else if (colIndex<=32767) {
                mv.visitIntInsn(Opcodes.SIPUSH, colIndex);
            } else {
                mv.visitLdcInsn(colIndex);
            }
        }

        mv.visitInsn(Opcodes.DASTORE);
    }

     */







}
