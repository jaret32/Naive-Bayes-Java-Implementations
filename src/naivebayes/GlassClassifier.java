/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package naivebayes;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;

/**
 *
 * @author Jaret
 */

public class GlassClassifier {
    
    // used for dicretization of continuous values
    private double[] maxes;
    private double[] mins;
    private int bins = 7;
    private double[] steps;
    
    private ArrayList<double[]> continuousData;
    private ArrayList<int[]> datasets;
    
    private int numClasses = 7;
    // holds attribute totals for calculating probabilities
    private int[][][] totals;
    // used to track fp, fn, tp, and tn
    private int[][] cMatrix;
    
    // keep track of class totals when training
    private int[] counts = new int[7];
    private int total = 0;
    
    
    public GlassClassifier() {
        maxes = new double[9];
        mins = new double[9];
        steps = new double[9];
        Arrays.fill(maxes, Double.MIN_VALUE);
        Arrays.fill(mins, Double.MAX_VALUE);
        continuousData = new ArrayList();
        datasets = new ArrayList();
    }
    
    // reads the data file and converts the rows into datasets
    public void processData(String filePath) throws FileNotFoundException {
        Scanner sc = new Scanner(new File(filePath));
        ArrayList<String> lines = new ArrayList();
        
        while (sc.hasNextLine()) lines.add(sc.nextLine());
        int rows = lines.size();
        
        for (int i = 0; i < rows; i++) {
            String[] parts = lines.get(i).split(",");
            double[] data = new double[parts.length - 1];
            
            // check for min and max value of each attribute and record
            for (int j = 1; j < parts.length; j++) {
                double val = Double.parseDouble(parts[j]);
                if (j < parts.length -1) {
                    if (val > maxes[j-1]) maxes[j-1] = val;
                    if (val < mins[j-1]) mins[j-1] = val;
                }
                data[j-1] = val;
            }
            continuousData.add(data);
        }
        
        // calculate a range to convert continuous data to discrete
        for (int i = 0; i < steps.length; i++) steps[i] = (maxes[i] - mins[i]) / bins;
        
        for (int i = 0; i < rows; i++) {
            double[] datapoint = continuousData.get(i);
            int[] data = new int[datapoint.length];
            // calculate which range it falls into and assign discrete value
            for (int j = 0; j < datapoint.length - 1; j++) {
                int k = 0;
                while ((k+1) * steps[j] + mins[j] < datapoint[j]) k++;
                data[j] = k;
            }
            data[datapoint.length-1] = (int) datapoint[datapoint.length-1];
            datasets.add(data);
        }
        // shuffle the dataset before 10-fold cv
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
    
    // trains the model with the given examples
    public void train(List<int[]> examples) {
        // reset totals
        Arrays.fill(counts, 0);
        total = 0;
        totals = new int[7][9][bins];
      
        int[] datapoint;
        for (int i = 0; i < examples.size(); i++) {
            datapoint = examples.get(i);
            // increment class totals
            int type = datapoint[9];
            counts[type-1]++;
            total++;
            // increment attribute totals
            for (int j = 0; j < datapoint.length - 1; j++) {
                totals[type-1][j][datapoint[j]]++;
            }
        }
    }
    
    // returns the predicted class for a given dataset
    public int classify(int[] observation) {
        double[] probabilities = new double[7];
        
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = (double)counts[i]/total;
        }
        
        for (int i = 0; i < observation.length - 1; i++) {
            for (int j = 0; j < probabilities.length; j++) {
                if (counts[j] == 0) probabilities[j] = 0;
                else probabilities[j] = probabilities[j] * ((double)totals[j][i][observation[i]]+1/counts[j]);
            }
        }
        
        double max = 0;
        int type = -1;
        
        for (int i = 0; i < probabilities.length; i++) {
            if (probabilities[i] > max) {
                max = probabilities[i];
                type = i;
            }
        }
        
        return type;
    }
    
    public void test(List<int[]> tests) {
        for (int[] test : tests) {
            int actual = test[9];
            int predicted = classify(test);
            if (predicted == -1);
            else cMatrix[actual-1][predicted]++;
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
        
        System.out.println("CONFUSION MATRIX FOR GLASS CLASSIFIER: \n");
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
