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


    public double[] costGradientW;
    public double[] costGradientB;
    
    public double[]inputs;
    public double[] notActivatedWeightenedInputs;
    public Layer(int nodesIn, int nodesOut)
    {
        this.nodesIn = nodesIn;
        this.nodesOut = nodesOut;
        weightsIn= new double[nodesIn * nodesOut];
        biases = new double[nodesOut];
            
        costGradientW = new double[nodesIn * nodesOut];
        costGradientB = new double[nodesOut];

        notActivatedWeightenedInputs= new double[nodesOut];
    }



    public void ApplyGradients(double learnRate)
    {
        for (int i = 0; i < weightsIn.Length; i++)
        {
            weightsIn[i] -= costGradientW[i] * learnRate;
        }
        for (int i = 0; i < biases.Length; i++)
        {
            biases[i] -= costGradientB[i] * learnRate;
        }

 
    }
    public void ClearGradients()
    {
        for (int i = 0; i < costGradientW.Length; i++)
        {
            costGradientW[i] = 0;
        }
        for (int i = 0; i < costGradientB.Length; i++)
        {
            costGradientB[i] = 0;
        }
    }
    public void UpdateGradients(double[] nodeValues)
    {
        int weightIndex = 0;
        for (int i = 0; i < nodesOut; i++)
        {
            for (int j = 0; j < nodesIn; j++)
            {
                double derivativeCostWeight = inputs[j] * nodeValues[i];
                costGradientW[weightIndex] += derivativeCostWeight;
                weightIndex++;
            }

        }
        for (int i = 0; i < biases.Length; i++)
        {
            double derivativeCostBias = 1 * nodeValues[i];
            costGradientB[i] += derivativeCostBias;
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
                newNodeValue += weightedInputDer * oldNodeValues[j];
                weightIndex++;
            }
            newNodeValue *= ActivationDerivative(notActivatedWeightenedInputs[i]);
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
            double activationDerivative = ActivationDerivative(notActivatedWeightenedInputs[i]);
            nodeValues[i] = costDerivative * activationDerivative;
        }
        return nodeValues;
        
    }
    public double [] CalculateLayer(double[] inputs)
    {
        double [] auxWeightendInputs= new double[this.nodesOut];
        notActivatedWeightenedInputs= new double[this.nodesOut];

        this.inputs = inputs;
        int weightIndex = 0;

        for (int i = 0; i < auxWeightendInputs.Length; i++)
        {
            for (int j = 0; j < inputs.Length; j++)
            {
                auxWeightendInputs[i] += inputs[j] * weightsIn[weightIndex];
                weightIndex++;
            }
            auxWeightendInputs[i] = auxWeightendInputs[i] + biases[i];
            notActivatedWeightenedInputs[i] = auxWeightendInputs[i];
            auxWeightendInputs[i] = ActivationFunction(auxWeightendInputs[i]);
        }
        this.weightedInputs = auxWeightendInputs;
        return auxWeightendInputs;
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

    public void CreateRandomWeights()
    {
        for (int i = 0; i < weightsIn.Length; i++)
        {
            weightsIn[i] = Random.Range(-0.1f, 0.1f) / (weightsIn.Length / 2);
        }
    }

    public void CreateBiases()
    {
        for (int i = 0; i < biases.Length; i++)
        {
            biases[i] = Random.Range(-0.01f, 0.01f);
        }
    }
    public void CreateRandomWeightGradients()
    {
        for (int i = 0; i < costGradientW.Length; i++)
        {
            costGradientW[i] = Random.Range(-1f, 1f);
        }
    }
}

