using System.Collections;
using System.Collections.Generic;
using System.Data;
using UnityEngine;

[System.Serializable]
public class NeuralNet
{
    public int numInputs;
    public int numHiddenLayers;
    public int numHiddenLayerSize;
    public int numOutputs;
    public data myDataset;
    [SerializeField]
    public Layer[] layers;


    public NeuralNet(int numInputs, int numHiddenLayers, int numHiddenLayerSize, int numOutputs)
    {
        this.numInputs = numInputs;
        this.numHiddenLayers = numHiddenLayers;
        this.numHiddenLayerSize = numHiddenLayerSize;
        this.numOutputs = numOutputs;

    }
    public void SetDataset(data myDataset)
    {
        this.myDataset = myDataset;
    }

    public void ConstructNet()
    {
        layers = new Layer[numHiddenLayers + 1];
        for (int i = 0; i < numHiddenLayers + 1; i++)
        {
            if (i == 0)
            {
                layers[i] = new Layer(numInputs, numHiddenLayerSize);
                continue;
            }
            if (i == numHiddenLayers)
            {
                layers[i] = new Layer(numHiddenLayerSize, numOutputs);
                continue;
            }
            layers[i] = new Layer(numHiddenLayerSize, numHiddenLayerSize);
        }

    }
    public void FillWeights()
    {
        for (int i = 0; i < numHiddenLayers + 1; i++)
        {
            layers[i].CreateRandomWeights();
        }

    }

    public double CostCalculation(data myData)
    {
        this.myDataset = myData;
        CalculateNet();
        Layer lastLayer = layers[layers.Length - 1];
        double cost = 0;
        for (int i = 0; i < lastLayer.weightedInputs.Length; i++)
        {
            cost += lastLayer.ReturnCost(lastLayer.weightedInputs[i], myDataset.outputs[i]);
        }
        return cost;
    }

    public void CalculateNet()
    {
        for (int i = 0; i < numHiddenLayers + 1; i++)
        {
            if (i == 0)
            {
                layers[i].CalculateLayer(myDataset.inputs);
                continue;
            }
            layers[i].CalculateLayer(layers[i - 1].weightedInputs);
        }
    }
    public void Train(data dataSet)
    {
        double h= 0.0001f;
        myDataset = dataSet;
        double originalCost= CostCalculation(myDataset);

        for (int i = 0; i < layers.Length; i++)
        {
            for (int j = 0; j < layers[i].weightsIn.Length; j++)
            {

                layers[i].weightsIn[j] += h;
                double deltaCost = CostCalculation(myDataset) - originalCost;
                layers[i].weightsIn[j] -= h;
                layers[i].weightGradients[j] = deltaCost / h;

            }
            for (int j = 0; j < layers[i].biases.Length; j++)
            {
                layers[i].biases[j] += h;
                double deltaCost = CostCalculation(myDataset) - originalCost;
                layers[i].biases[j] -= h;
                layers[i].biasGradients[j] = deltaCost / h;
            }
           

        }
        ApplyAllGradients(0.1f);

    }
    public void ApplyAllGradients(double learnRate)
    {
        for (int i = 0; i < layers.Length; i++)
        {
            layers[i].ApplyGradients(learnRate);
        }
    }




}
[System.Serializable]
public struct data
{
    public double[] inputs;
    public double[] outputs;
    public data(double[] inputs, double[] outputs)
    {
        this.inputs = inputs;
        this.outputs = outputs;
    }
}