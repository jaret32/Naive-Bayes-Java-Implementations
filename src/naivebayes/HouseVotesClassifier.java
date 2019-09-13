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

public class HouseVotesClassifier {
    
    private int numClasses = 2;
    private ArrayList<int[]> datasets;
    
    private int[][][] totals;
    private int[][] cMatrix;
    
    // keep track of class totals when training
    private int[] counts = new int[numClasses];
    private int total;
    
    
    public HouseVotesClassifier() {
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
            int[] data = new int[parts.length];
            
            if (parts[0].equals("democrat")) data[0] = 0;
            else data[0] = 1;
            
            for (int j = 1; j < parts.length; j++) {
                if (parts[j].equals("n")) data[j] = 0;
                else if (parts[j].equals("y")) data[j] = 1;
                else data[j] = 2;
            }
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
        for (int i = 1; i < numAttributes + 1; i++) list.add(i);
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
        totals = new int[2][16][3];
      
        int[] datapoint;
        for (int i = 0; i < examples.size(); i++) {
            datapoint = examples.get(i);
            // increment class total
            int party = datapoint[0];
            counts[party]++;
            total++;
            // increment attribute totals
            for (int j = 1; j < datapoint.length; j++) {
                totals[party][j-1][datapoint[j]]++;
            }
        }
    }
    
    // returns the predicted class for a given dataset
    public int classify(int[] observation) {
        double[] probabilities = new double[2];
        
        // initialize probabilites to class probabilites
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = (double)counts[i]/total;
        }
        
        // for each attribute multiply previous probability by attribute probability
        for (int i = 1; i < observation.length; i++) {
            for (int j = 0; j < probabilities.length; j++) {
                probabilities[j] = probabilities[j] * ((double)totals[j][i-1][observation[i]]+1/counts[j]);
            }
        }
        
        // return argmax of probabilities
        if (probabilities[0] > probabilities[1]) return 0;
        else if (probabilities[1] > probabilities[0]) return 1;
        // if they have the same probability return which ever one occurs more in training data
        else return counts[1] > counts[0] ? 1 : 0;
    }
    
    // tests the model with given test set and builds confusion matrix
    public void test(List<int[]> tests) {
        for (int[] test : tests) {
            int actual = test[0];
            int predicted = classify(test);
            cMatrix[actual][predicted]++;
        }
    }
    
    // creates 10 folds for cross validation and trains/tests each iteration
    public void runCrossValidation() {
        // create new confusion matrix
        cMatrix = new int[numClasses][numClasses];
        // get size of each partition
        int partitionSize = (int)((double)datasets.size() / 10);
        
        // iterate over 10 folds
        for (int fold = 1; fold <= 10; fold++) {
            int end = partitionSize * fold;
            int start = end - partitionSize;
            // create training examples by splicing out test sets
            List<int[]> examples = new ArrayList(datasets.subList(0, start));
            examples.addAll(datasets.subList(end + 1, datasets.size()));
            // splice out test sets
            List<int[]> tests = datasets.subList(start, end + 1);
            // train the model with the examples and test the test set
            this.train(examples);
            this.test(tests);
        }
        
        // prints confusion matrix
        System.out.println("CONFUSION MATRIX FOR HOUSE VOTES CLASSIFIER: \n");
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
