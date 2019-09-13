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

/**
 *
 * @author Jaret
 */
class SoybeanClassifier {
    
    private int numClasses = 4;
    private ArrayList<int[]> datasets;
    
    // holds attribute totals for calculating probabilities
    private int[][][] totals;
    // used to track fp, fn, tp, and tn
    private int[][] cMatrix;
    
    // keep track of class totals when training
    private int[] counts = new int[numClasses];
    private int total = 0;
    
    
    public SoybeanClassifier() {
        datasets = new ArrayList();
        cMatrix = new int[numClasses][numClasses];
    }
    
    // reads the data file and converts the rows into datasets
    public void processData(String filePath) throws FileNotFoundException {
        Scanner sc = new Scanner(new File(filePath));
        ArrayList<String> lines = new ArrayList();
        
        while (sc.hasNextLine()) lines.add(sc.nextLine());
        int rows = lines.size();
        
        for (int i = 0; i < rows; i++) {
            String[] parts = lines.get(i).split(",");
            int[] data = new int[parts.length];
            
            for (int j = 0; j < parts.length - 1; j++) data[j] = Integer.parseInt(parts[j]);
            
            String type = parts[parts.length-1];
            int t = -1;
            
            if (type.equals("D1")) t = 0;
            else if (type.equals("D2")) t = 1;
            else if (type.equals("D3")) t = 2;
            else t = 3;
            
            data[parts.length-1] = t;
            datasets.add(data);
        }
        // shuffle the dataset before 10-fold cv
        Collections.shuffle(datasets);
    }
    
    // trains the model with the given examples
    public void train(List<int[]> examples) {
        // reset totals
        Arrays.fill(counts, 0);
        total = 0;
        totals = new int[4][35][7];
      
        int[] datapoint;
        for (int i = 0; i < examples.size(); i++) {
            datapoint = examples.get(i);
            // increment class total
            int type = datapoint[35];
            counts[type]++;
            total++;
            
            // increment attribute totals
            for (int j = 0; j < datapoint.length - 1; j++) {
                totals[type][j][datapoint[j]]++;
            }
        }
    }
    
    // returns the predicted class for a given dataset
    public int classify(int[] observation) {
        double[] probabilities = new double[4];
        
        // initialize probabilites to class probabilites
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = (double)counts[i]/total;
        }
        
        // for each attribute multiply previous probability by attribute probability
        for (int i = 0; i < observation.length - 1; i++) {
            for (int j = 0; j < probabilities.length; j++) {
                probabilities[j] = probabilities[j] * ((double)totals[j][i][observation[i]]/counts[j]);
            }
        }
        
        // find and return argmax of the probabilities
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
    
    // tests the model with given test set and builds confusion matrix
    public void test(List<int[]> tests) {
        for (int[] test : tests) {
            int actual = test[35];
            int predicted = classify(test);
            cMatrix[actual][predicted]++;
        }
    }
    
    // creates 10 folds for cross validation and trains/tests each iteration
    public void runCrossValidation() {
        // get size of each partition
        int partitionSize = (int)((double)datasets.size() / 10);
        // iterate over 10 folds
        for (int fold = 1; fold <= 10; fold++) {
            int end = partitionSize * fold;
            int start = end - partitionSize;
            // create training examples by splicing out test sets
            List<int[]> examples = datasets.subList(0, start);
            examples.addAll(datasets.subList(end + 1, datasets.size()));
            // splice out test sets
            List<int[]> tests = datasets.subList(start, end + 1);
            // train the model with the examples and test the test set
            this.train(examples);
            this.test(tests);
        }
        // prints confusion matrix
        System.out.println("CONFUSION MATRIX FOR SOYBEAN CLASSIFIER: \n");
        for (int i = 0; i < cMatrix.length; i++) {
            for (int j = 0; j < cMatrix[i].length; j++) {
                System.out.print(String.format("%20s", cMatrix[i][j]));
            }
            System.out.println();
        }
        System.out.println();
        //System.out.println(Arrays.deepToString(cMatrix).replace("], ", "]\n").replace("[[", "[").replace("]]", "]"));
    }
    
}
