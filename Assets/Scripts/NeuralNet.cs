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

    public double learningRate;

    public NeuralNet(int numInputs, int numHiddenLayers, int numHiddenLayerSize, int numOutputs, double learningRate)
    {
        this.numInputs = numInputs;
        this.numHiddenLayers = numHiddenLayers;
        this.numHiddenLayerSize = numHiddenLayerSize;
        this.numOutputs = numOutputs;
        this.learningRate = learningRate;
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
            layers[i].CreateBiases();
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
    public void Train(data[] dataSet)
    {
        for (int i = 0; i < dataSet.Length; i++)
        {
            UpdateAllGradients(dataSet[i]);
        }
        ApplyAllGradients(learningRate / dataSet.Length);

        ClearAllGradients();


    }
    public void UpdateAllGradients(data dataPoint)
    {
        CalculateNet();

        Layer outputLayer = layers[layers.Length-1];
        double[] nodeValues = outputLayer.CalculateOutputLayerNodeVal(dataPoint.outputs);
        outputLayer.UpdateGradients(nodeValues);
        for (int i = layers.Length - 2; i >= 0; i--)
        {
            Layer hiddenLayer = layers[layers.Length - 2];
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeVal(outputLayer, nodeValues);
            hiddenLayer.UpdateGradients(nodeValues);
        }

    }
    public void ClearAllGradients()
    {
          for (int i = 0; i < layers.Length; i++)
            {
                layers[i].ClearGradients();
            }
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