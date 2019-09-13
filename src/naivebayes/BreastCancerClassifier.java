/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package naivebayes;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;

/**
 *
 * @author Jaret
 */

public class BreastCancerClassifier {
    
    private ArrayList<int[]> datasets;
    private int[][][] totals;
    private int[][] cMatrix;
    
    private int numClasses = 2;
    // keep track of class totals when training
    private int numBenign;
    private int numMalignant;
    
    
    public BreastCancerClassifier() {
        datasets = new ArrayList();
        cMatrix = new int[2][2];
    }
    
    public void processData(String filePath) throws FileNotFoundException {
        Scanner sc = new Scanner(new File(filePath));
        ArrayList<String> lines = new ArrayList();
        
        while (sc.hasNextLine()) lines.add(sc.nextLine());
        int rows = lines.size();
        
        for (int i = 0; i < rows; i++) {
            String[] parts = lines.get(i).split(",");
            int[] data = new int[parts.length - 1];
            for (int j = 1; j < parts.length; j++) {
                if (parts[j].equals("?")) data[j-1] = 0;
                else data[j-1] = Integer.parseInt(parts[j]);
            }
            datasets.add(data);
        }
        Collections.shuffle(datasets);
    }
    
    // shuffle 10 percent of attributes
    public void shuffleData() {
        int numExamples = datasets.size();
        int numAttributes = datasets.get(0).length - 1;
        int numCols = (int)Math.ceil(0.1 * numAttributes);
        ArrayList<Integer> list = new ArrayList();
        for (int i = 0; i < numAttributes; i++) list.add(i);
        Collections.shuffle(list);
        for (int i = 0; i < numCols; i++) {
            int col = list.get(i);
            for (int j = 0; j < numExamples; j++) {
                int[] randoms = ThreadLocalRandom.current().ints(numExamples, 0, numExamples).toArray();
                int temp = datasets.get(j)[col];
                datasets.get(j)[col] = datasets.get(randoms[j])[col];
                datasets.get(randoms[j])[col] = temp;
            }
        }
    }
    
    // rework 10-fold cross validation
    public void train(List<int[]> examples) {
        // reset totals
        numBenign = 0;
        numMalignant = 0;
        totals = new int[2][9][10];
      
        int[] datapoint;
        for (int i = 0; i < examples.size(); i++) {
            datapoint = examples.get(i);
            
            int type = datapoint[9];
            if (type == 2) numBenign++;
            else numMalignant++;
            
            for (int j = 0; j < datapoint.length - 1; j++) {
                // if we have a missing value, don't count it
                if (datapoint[j] == 0) continue;
                totals[type/2-1][j][datapoint[j]-1]++;
            }
        }
    }
    
    // returns the predicted class for a given dataset
    public int classify(int[] observation) {
        double pBenign = (double)numBenign/(numBenign + numMalignant);
        double pMalignant = (double)numMalignant/(numBenign + numMalignant);
        
        for (int i = 0; i < observation.length - 1; i++) {
            if (observation[i] != 0) {
                pBenign = pBenign * ((double)totals[0][i][observation[i]-1]/numBenign);
                pMalignant = pMalignant * ((double)totals[1][i][observation[i]-1]/numMalignant);
            }
        }
        
        if (pBenign > pMalignant) return 2;
        else if (pMalignant > pBenign) return 4;
        // if they have the same probability return which ever one occurs more in training data
        else return numBenign > numMalignant ? 2 : 4;
    }
    
    public void test(List<int[]> tests) {
        
        for (int[] test : tests) {
            int actual = test[9];
            int predicted = classify(test);
            cMatrix[actual/2-1][predicted/2-1]++;
        }
    }
    
    public void runCrossValidation() {
        // create new confusion matrix
        cMatrix = new int[numClasses][numClasses];
        
        int partitionSize = (int)((double)datasets.size() / 10);
        
        for (int fold = 1; fold <= 10; fold++) {
            int end = partitionSize * fold;
            int start = end - partitionSize;
            
            List<int[]> examples = new ArrayList(datasets.subList(0, start));
            examples.addAll(datasets.subList(end + 1, datasets.size()));
            
            List<int[]> tests = datasets.subList(start, end + 1);
            
            this.train(examples);
            this.test(tests);
        }
        
        System.out.println("CONFUSION MATRIX FOR BREAST CANCER CLASSIFIER: \n");
        for (int i = 0; i < cMatrix.length; i++) {
            for (int j = 0; j < cMatrix[i].length; j++) {
                System.out.print(String.format("%20s", cMatrix[i][j]));
            }
            System.out.println();
        }
        System.out.println();
        System.out.println("Accuracy: " + Loss.calculateAccuracy(cMatrix));
        System.out.println("Macro-Average Precision: " + Loss.calculatePrecision(cMatrix));
        System.out.println("Macro-Average Recall: " + Loss.calculateRecall(cMatrix));
        System.out.println();
    }
    
}
