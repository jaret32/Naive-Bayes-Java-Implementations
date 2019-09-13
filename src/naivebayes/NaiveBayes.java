/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package naivebayes;

import java.io.FileNotFoundException;

/**
 *
 * @author Jaret
 */
public class NaiveBayes {

    /**
     * @param args the command line arguments
     */
    
    public static void main(String[] args) {
        BreastCancerClassifier breastCancerClassifier = new BreastCancerClassifier();
        GlassClassifier glassClassifier = new GlassClassifier();
        IrisClassifier irisClassifier = new IrisClassifier();
        SoybeanClassifier soybeanClassifier = new SoybeanClassifier();
        HouseVotesClassifier houseVotesClassifier = new HouseVotesClassifier();
        
        try {
            breastCancerClassifier.processData("breast-cancer-wisconsin.data");
            glassClassifier.processData("glass.data");
            irisClassifier.processData("iris.data");
            soybeanClassifier.processData("soybean-small.data");
            houseVotesClassifier.processData("house-votes-84.data");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return;
        }
        
        breastCancerClassifier.runCrossValidation();
        glassClassifier.runCrossValidation();
        irisClassifier.runCrossValidation();
        soybeanClassifier.runCrossValidation();
        houseVotesClassifier.runCrossValidation();
    }
    
}