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

    public double[] costGradientW;
    public double[] costGradientB;
    
    public double[]inputs;


    public Layer(int nodesIn, int nodesOut)
    {
        this.nodesIn = nodesIn;
        this.nodesOut = nodesOut;
        weightsIn= new double[nodesIn * nodesOut];
        biases = new double[nodesOut];


        weightGradients = new double[nodesIn * nodesOut];
        biasGradients = new double[nodesOut];
            
            
        costGradientW = new double[nodesIn * nodesOut];
        costGradientB = new double[nodesOut];

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
            biases[i] = 0;
        } 
    }
    public void CreateRandomWeightGradients()
    {
        for (int i = 0; i < weightGradients.Length; i++)
        {
            weightGradients[i] = Random.Range(-1f, 1f);
        }
    }
    public void ClearGradients()
    {
        for (int i = 0; i < weightGradients.Length; i++)
        {
            weightGradients[i] = 0;
        }
        for (int i = 0; i < biasGradients.Length; i++)
        {
            biasGradients[i] = 0;
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
    public void UpdateGradients(double[] nodeValues)
    {
        int weightIndex = 0;
        for (int i = 0; i < this.nodesOut; i++)
        {
            for (int j = 0; j < this.nodesIn; j++)
            {
                double derivativeCostWeight= inputs[j] * nodeValues[i];
                weightGradients[weightIndex] += derivativeCostWeight;
            }
            double derivativeCostBias = 1 * nodeValues[i];
            biasGradients[i] += derivativeCostBias;
        }

    }

    public double[] CalculateHiddenLayerNodeVal(Layer oldLayer, double[] oldNodeValues)
    {
        double[] newNodeValues = new double[nodesOut];
        int weightIndex = 0;
        for (int i = 0; i < newNodeValues.Length; i++)
        {
            double newNodeValue = 0;
            for (int j = 0; j < oldNodeValues.Length; j++)
            {
                double weightedInputDer = oldLayer.weightsIn[weightIndex];
                newNodeValue= weightedInputDer * oldNodeValues[j];
            }
            newNodeValue *= ActivationDerivative(weightedInputs[i]);
            newNodeValues[i] = newNodeValue;
        }
        return newNodeValues;
    }

    public double [] CalculateOutputLayerNodeVal(double[] expectedOutputs)
    {
        double[] nodeValues = new double[expectedOutputs.Length];
        for (int i = 0; i < nodeValues.Length; i++)
        {
            double costDerivative = CostDerivative(weightedInputs[i], expectedOutputs[i]);
            double activationDerivative = ActivationDerivative(weightedInputs[i]);
            nodeValues[i] = costDerivative * activationDerivative;
        }
        return nodeValues;
        
    }
    public double [] CalculateLayer(double[] inputs)
    {
        double [] weightendInputs= new double[this.nodesOut];
        this.inputs = inputs;
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


    public double CostDerivative(double output, double expectedOutput)
    {
        return 2 * (output - expectedOutput);
    }
    public double ActivationDerivative(double input)
    {
        return 1 - System.Math.Pow(System.Math.Tanh(input), 2);

    }
    
}

