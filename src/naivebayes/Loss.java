/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package naivebayes;

/**
 *
 * @author Jaret
 */
public class Loss {
    
    public static double calculateAccuracy(int[][] cMatrix) {
        int tp = 0;
        int total = 0;
        for (int i = 0; i < cMatrix.length; i++) {
            for (int j = 0; j < cMatrix[i].length; j++) {
                if (i == j) tp += cMatrix[i][j];
                total+= cMatrix[i][j];
            }
        }
        return (double)tp/total;
    }
    
    public static double calculatePrecision(int[][] cMatrix) {
        double precision = 0;
        for (int i = 0; i < cMatrix.length; i++) {
            int tp = 0;
            int total = 0;
            for (int j = 0; j < cMatrix[i].length; j++) {
                if (i == j) {
                    tp += cMatrix[j][i];
                    total += cMatrix[j][i];
                } else {
                    total +=cMatrix[j][i];
                }
                
            }
            if (tp !=0 && total != 0) precision += (double)tp/total;
        }
        return precision/cMatrix.length;
    }
    
    public static double calculateRecall(int[][] cMatrix) {
        double recall = 0;
        for (int i = 0; i < cMatrix.length; i++) {
            int tp = 0;
            int total = 0;
            for (int j = 0; j < cMatrix[i].length; j++) {
                if (i == j) {
                    tp += cMatrix[i][j];
                    total += cMatrix[i][j];
                } else {
                    total +=cMatrix[i][j];
                }
                
            }
            if (tp !=0 && total != 0) recall += (double)tp/total;
        }
        return recall/cMatrix.length;
    }
    
}
