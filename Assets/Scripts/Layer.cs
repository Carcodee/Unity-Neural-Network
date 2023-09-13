using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[System.Serializable]
public class Layer 
{
    public double [] weightsIn;

    public int nodesIn;
    public int nodesOut;

    public double[] biases;
    public double[] weightedInputs;

    public double[] weightGradients;
    public double[] biasGradients;

    public Layer(int nodesIn, int nodesOut)
    {
        this.nodesIn = nodesIn;
        this.nodesOut = nodesOut;
        weightsIn= new double[nodesIn * nodesOut];
        biases = new double[nodesOut];


        weightGradients = new double[nodesIn * nodesOut];
        biasGradients = new double[nodesOut];
    }
    public void CreateRandomWeights()
    {
        for (int i = 0; i < weightsIn.Length; i++)
        {
            weightsIn[i] = Random.Range(-1f, 1f)/ (weightsIn.Length/2);
        }
    }

    public void CreateBiases()
    {
        for (int i = 0; i < biases.Length; i++)
        {
            biases[i] =0;
        } 
    }
    public void CreateRandomWeightGradients()
    {
        for (int i = 0; i < weightGradients.Length; i++)
        {
            weightGradients[i] = Random.Range(-1f, 1f);
        }
    }

    public void ApplyGradients(double learnRate)
    {
        for (int i = 0; i < biases.Length; i++)
        {
            biases[i] -= biasGradients[i] * learnRate;
        }

        for (int i = 0; i < weightsIn.Length; i++)
        {
            weightsIn[i] -= weightGradients[i] * learnRate;
        }
    }

    public double [] CalculateLayer(double[] inputs)
    {
        double [] weightendInputs= new double[this.nodesOut];
        CreateBiases();
        int weightIndex = 0;

        for (int i = 0; i < weightendInputs.Length; i++)
        {
            for (int j = 0; j < inputs.Length; j++)
            {
                weightendInputs[i] += inputs[j] * weightsIn[weightIndex] + biases[i];
                weightIndex++;
            }
            weightendInputs[i] = ActivationFunction(weightendInputs[i]);
        }
        this.weightedInputs = weightendInputs;
        return weightendInputs;
    }
    double ActivationFunction(double input)
    {
        return System.Math.Tanh(input);
    }

    public double ReturnCost(double output, double expectedOutput)
    {
        double error= output - expectedOutput;
        return System.Math.Pow(error, 2);
    }
    
}

