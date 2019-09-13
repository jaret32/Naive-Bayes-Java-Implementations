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

public class GlassClassifier {
    
    private double[] maxes;
    private double[] mins;
    private int bins = 7;
    private double[] steps;
    
    private ArrayList<double[]> continuousData;
    private ArrayList<int[]> datasets;
    
    private int[][][] totals;
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
        cMatrix = new int[7][7];
    }
    
    public void processData(String filePath) throws FileNotFoundException {
        Scanner sc = new Scanner(new File(filePath));
        ArrayList<String> lines = new ArrayList();
        
        while (sc.hasNextLine()) lines.add(sc.nextLine());
        int rows = lines.size();
        
        for (int i = 0; i < rows; i++) {
            String[] parts = lines.get(i).split(",");
            double[] data = new double[parts.length - 1];
            
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
        
        for (int i = 0; i < steps.length; i++) steps[i] = (maxes[i] - mins[i]) / bins;
        
        for (int i = 0; i < rows; i++) {
            double[] datapoint = continuousData.get(i);
            int[] data = new int[datapoint.length];
            
            for (int j = 0; j < datapoint.length - 1; j++) {
                int k = 0;
                while ((k+1) * steps[j] + mins[j] < datapoint[j]) k++;
                data[j] = k;
            }
            data[datapoint.length-1] = (int) datapoint[datapoint.length-1];
            datasets.add(data);
        }
        Collections.shuffle(datasets);
    }
    
    // rework 10-fold cross validation
    public void train(List<int[]> examples) {
        // reset totals
        Arrays.fill(counts, 0);
        total = 0;
        totals = new int[7][9][bins];
      
        int[] datapoint;
        for (int i = 0; i < examples.size(); i++) {
            datapoint = examples.get(i);
            
            int type = datapoint[9];
            counts[type-1]++;
            total++;
            
            for (int j = 0; j < datapoint.length - 1; j++) {
                totals[type-1][j][datapoint[j]]++;
            }
        }
    }
    
    public int classify(int[] observation) {
        double[] probabilities = new double[7];
        
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] = (double)counts[i]/total;
        }
        
        for (int i = 0; i < observation.length - 1; i++) {
            for (int j = 0; j < probabilities.length; j++) {
                probabilities[j] = probabilities[j] * ((double)totals[j][i][observation[i]]/counts[j]);
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
            cMatrix[actual-1][predicted]++;
        }
    }
    
    public void runCrossValidation() {
        int partitionSize = (int)((double)datasets.size() / 10);
        
        for (int fold = 1; fold <= 10; fold++) {
            int end = partitionSize * fold;
            int start = end - partitionSize;
            
            List<int[]> examples = datasets.subList(0, start);
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
    }
    
}
