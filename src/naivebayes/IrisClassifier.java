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
public class IrisClassifier {
    
    private double[] maxes;
    private double[] mins;
    private int bins = 3;
    private double[] steps;
    
    private ArrayList<double[]> continuousData;
    private ArrayList<int[]> datasets;
    
    private int[][][] totals;
    private int[][] cMatrix;
    
    // keep track of class totals when training
    private int[] counts = new int[3];
    private int total = 0;
    
    
    public IrisClassifier() {
        maxes = new double[4];
        mins = new double[4];
        steps = new double[4];
        Arrays.fill(maxes, Double.MIN_VALUE);
        Arrays.fill(mins, Double.MAX_VALUE);
        continuousData = new ArrayList();
        datasets = new ArrayList();
        cMatrix = new int[3][3];
    }
    
    public void processData(String filePath) throws FileNotFoundException {
        Scanner sc = new Scanner(new File(filePath));
        ArrayList<String> lines = new ArrayList();
        
        while (sc.hasNextLine()) lines.add(sc.nextLine());
        int rows = lines.size();
        
        for (int i = 0; i < rows; i++) {
            String[] parts = lines.get(i).split(",");
            if (parts.length == 1) continue;
            double[] data = new double[parts.length];
            
            for (int j = 0; j < parts.length - 1; j++) {
                double val = Double.parseDouble(parts[j]);
                if (val > maxes[j]) maxes[j] = val;
                if (val < mins[j]) mins[j] = val;
                data[j] = val;
            }
            
            String type = parts[parts.length - 1];
            int t = -1;
            
            if (type.equals("Iris-setosa")) t = 0;
            else if (type.equals("Iris-versicolor")) t = 1;
            else t = 2;
            
            data[parts.length - 1] = t;
            continuousData.add(data);
        }
        
        for (int i = 0; i < steps.length; i++) steps[i] = (maxes[i] - mins[i]) / bins;
        
        for (int i = 0; i < continuousData.size(); i++) {
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
        totals = new int[3][4][bins];
      
        int[] datapoint;
        for (int i = 0; i < examples.size(); i++) {
            datapoint = examples.get(i);
            //System.out.println(Arrays.toString(datapoint));
            int type = datapoint[4];
            counts[type]++;
            total++;
            
            for (int j = 0; j < datapoint.length - 1; j++) {
                totals[type][j][datapoint[j]]++;
            }
        }
    }
    
    public int classify(int[] observation) {
        double[] probabilities = new double[3];
        
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
        
        //System.out.println(Arrays.toString(probabilities));
        
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
            int actual = test[4];
            int predicted = classify(test);
            cMatrix[actual][predicted]++;
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
        
        System.out.println("CONFUSION MATRIX FOR IRIS CLASSIFIER: \n");
        for (int i = 0; i < cMatrix.length; i++) {
            for (int j = 0; j < cMatrix[i].length; j++) {
                System.out.print(String.format("%20s", cMatrix[i][j]));
            }
            System.out.println();
        }
        System.out.println();
    }
    
}
