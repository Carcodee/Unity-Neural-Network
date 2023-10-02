using System;
using System.Collections;
using System.Collections.Generic;
using System.Dynamic;
using TMPro;
using UnityEditor;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UIElements;

public class NeuralNetController : MonoBehaviour
{
    [SerializeField]
    private Camera cameraMain;

    [HideInInspector] public GameObject inputPrefab;
    [HideInInspector] public GameObject hiddenLayerPrefab;
    [HideInInspector] public GameObject outputPrefab;


    [Header("NET VALUES")]
    [Range(1,20)]
    public int numInputs;
    [Range(1, 20)]
    public int numHiddenLayers;
    [Range(1, 20)]
    public int numHiddenLayerSize;
    [Range(1, 20)]
    public int numOutputs;

    private int lastNumInputs;
    private int lastNumHiddenLayers;
    private int lastNumHiddenLayerSize;
    private int lastNumOutputs;

    [Header("NET")]
    private GameObject[] inputs;
    private GameObject[] hiddenLayerSizeY;
    private GameObject[] outputs;
    private bool dynamicNet = false;
    private bool backPropTrain = false;
    private bool gradientDescentTrain = false;

    [Header("Debug")]
    [HideInInspector] public Transform spawnPointsInputs;
    [HideInInspector] public Transform spawnPointsHiddenLayersX;
    [HideInInspector] public Transform spawnPointsHiddenLayerSizeY;
    [HideInInspector] public Transform spawnPointsOutputs;
    [HideInInspector] public TextMeshProUGUI cost;
    public TextMeshProUGUI inputCol;
    public TextMeshProUGUI outputCol;
    private hiddenLayer[] allMyHiddenLayers;


    [Header("Neural Net Data")]
    [Range(1, 500)]
    public int dataTrainSize;
    [Range(1, 100)]
    public int dataTestSize;
    [Range(1, 5)]
    public int batches = 1;
    [Range(0, 10)]
    public double learnRate = 0;
    //public Batch[] myBatches;

    [SerializeField]
    public data[] dataTrain;
    [SerializeField]
    public data[] dataTest;

    public NeuralNet myNet;
    private int dataIndex = 0;
    private int dataTestIndex = 0;

    [Header("Animation")]
    [SerializeField]
    float speedAnim;

    private float onTrainTimerColorLerp;

    void Start()
    {


        CreateNetUI(ref numInputs,ref numHiddenLayers,ref numHiddenLayerSize,ref numOutputs);
        CreateBackendNet();
        lastNumInputs = numInputs;
        lastNumHiddenLayers = numHiddenLayers;
        lastNumHiddenLayerSize = numHiddenLayerSize;
        lastNumOutputs = numOutputs;

}

    void Update()
    {
        if (dynamicNet)
        {
            OnValuesChange();
        }
        if (backPropTrain)
        {
            onTrainTimerColorLerp += Time.deltaTime * 0.1f;
            CalculateCost();
            myNet.Train(dataTrain,ref dataIndex);

        }
        if (gradientDescentTrain)
        {
            onTrainTimerColorLerp += Time.deltaTime * 0.1f;
            CalculateCost();
            myNet.GradientTrain(dataTrain);
            dataIndex++;
            if (dataIndex == dataTrain.Length)
            {
                dataIndex = 0;
            }

        }
        //lines UI

        if (backPropTrain)
        {
            for (int i = 0; i < allMyHiddenLayers.Length; i++)
            {
                SetOutputCols(ref allMyHiddenLayers[i].hiddenLayerSizeY, i, dataTrain[dataIndex].inputs);
            }
        }
        if (gradientDescentTrain)
        {
            for (int i = 0; i < allMyHiddenLayers.Length; i++)
            {
                SetOutputCols(ref allMyHiddenLayers[i].hiddenLayerSizeY, i, dataTrain[dataIndex].inputs);
            }
        }

    }
    private void FixedUpdate()
    {
        moveLines(ref inputs, ref allMyHiddenLayers[0].hiddenLayerSizeY, 0);

        for (int i = 0; i < allMyHiddenLayers.Length; i++)
        {
            if (i + 1 == allMyHiddenLayers.Length)
            {
                break;
            }
            moveLines(ref allMyHiddenLayers[i].hiddenLayerSizeY, ref allMyHiddenLayers[i + 1].hiddenLayerSizeY, i + 1);

        }
        moveLines(ref allMyHiddenLayers[allMyHiddenLayers.Length - 1].hiddenLayerSizeY, ref outputs, allMyHiddenLayers.Length);

    }
    public void TestNet()
    {
        gradientDescentTrain = false;
        backPropTrain = false;
        myNet.SetDataset(dataTest[dataTestIndex]);
        myNet.CalculateNet();
        //lines UI
        for (int i = 0; i < allMyHiddenLayers.Length; i++)
        {
            SetOutputCols(ref allMyHiddenLayers[i].hiddenLayerSizeY, i, dataTest[dataTestIndex].inputs);
        }
        dataTestIndex++;
        if (dataTestIndex==dataTest.Length-1)
        {
            dataTestIndex = 0;
        }
    }

    private void CreateBackendNet()
    {
        //backend

        //dataset


        CreateData();

        myNet = new NeuralNet(numInputs, numHiddenLayers, numHiddenLayerSize, numOutputs, learnRate);
        myNet.ConstructNet();
        myNet.SetNetActivations(ActivationType.Tanh);
        myNet.SetDataset(dataTrain[dataIndex]);
        myNet.FillWeights();
        myNet.CalculateNet();

    }

    void CreateData()
    {
        //myBatches = new Batch[batches];
        //myBatches[i] = new Batch(dataTrain);

        dataTrain = new data[dataTrainSize];
        dataTest = new data[dataTestSize];

        for (int j = 0; j < dataTrain.Length; j++)
            {
                double[] outputs = new double[numOutputs];
                double[] inputs = new double[numInputs];

                inputs[0] =UnityEngine.Random.Range(0f, 1f);
                inputs[1] = UnityEngine.Random.Range(0f, 1f);
                inputs[2] = UnityEngine.Random.Range(0f, 1f);

                outputs[0] = inputs[0];
                outputs[1] = inputs[1];
                outputs[2] = inputs[2];

                dataTrain[j] = new data(inputs, outputs);
            }

        for (int j = 0; j < dataTest.Length; j++)
        {
            double[] outputs = new double[numOutputs];
            double[] inputs = new double[numInputs];

            inputs[0] = UnityEngine.Random.Range(0f, 1f);
            inputs[1] = UnityEngine.Random.Range(0f, 1f);
            inputs[2] = UnityEngine.Random.Range(0f, 1f);

            outputs[0] = inputs[0];
            outputs[1] = inputs[1];
            outputs[2] = inputs[2];
            dataTest[j] = new data(inputs, outputs);
        }

    }
    #region screen
    public void Train()
    {
        onTrainTimerColorLerp= 0;
        backPropTrain = !backPropTrain;
        gradientDescentTrain=false;
    }
    public void GradientTrain()
    {
        onTrainTimerColorLerp = 0;
        gradientDescentTrain = !gradientDescentTrain;
        backPropTrain = false;

    }
    public void CalculateCost()
    {
        cost.text = myNet.TotalCost(dataTrain).ToString();
    }
    public void DynamicNet()
    {
        dynamicNet = !dynamicNet;
    }
    #endregion



    #region UI

    private void CreateNetUI(ref int numInputs,ref int numHiddenLayersX,ref int numHiddenLayerSize,ref int numOutputs)
    {
        inputs = new GameObject[numInputs];
        hiddenLayerSizeY=new GameObject[numHiddenLayerSize];
        outputs = new GameObject[numOutputs];
        numHiddenLayers = numHiddenLayersX;

        //objects UI
        CreateLayers(inputPrefab,ref inputs, spawnPointsInputs);
        allMyHiddenLayers= CreateHiddenLayers(hiddenLayerPrefab, ref hiddenLayerSizeY, spawnPointsHiddenLayerSizeY, ref numHiddenLayers);
        CreateLayers(outputPrefab,ref outputs, spawnPointsOutputs);

        //lines UI
        CreateConnections(ref inputs,ref allMyHiddenLayers[0].hiddenLayerSizeY);
        for (int i = 0; i < allMyHiddenLayers.Length; i++)
        {
            if (i + 1 == allMyHiddenLayers.Length)
            {
                break;
            }
            CreateConnections(ref allMyHiddenLayers[i].hiddenLayerSizeY,ref allMyHiddenLayers[i + 1].hiddenLayerSizeY);

        }
        CreateConnections(ref allMyHiddenLayers[allMyHiddenLayers.Length-1].hiddenLayerSizeY,ref outputs);



    }


    private void CreateLayers(GameObject prefab,ref GameObject[] array, Transform position)
    {
        float segment=(cameraMain.orthographicSize*2) - 2;
        float offsetY = segment / (array.Length+1);

        Vector3 currentOffset = new Vector3(0, 0, 0);
        for (int i = 0; i < array.Length; i++)
        {

                currentOffset.y += offsetY;
                array[i] = Instantiate(prefab, position.position + currentOffset, Quaternion.identity, position);
            
        }
    }  
    
    private hiddenLayer [] CreateHiddenLayers(GameObject prefab, ref GameObject[] hiddenLayerSizeY, Transform position, ref int numHiddenLayerSize)
    {
        hiddenLayer[] allMyHiddenLayers=new hiddenLayer[numHiddenLayerSize];

        List<GameObject> currentLayer=new List<GameObject>();


        float segmentX = (4 * 2);
        float segmentY = (cameraMain.orthographicSize * 2) - 2;

        float offsetX = segmentX / (numHiddenLayerSize + 1);
        float offsetY = segmentY / (hiddenLayerSizeY.Length + 1);

        Vector3 currentOffset = new Vector3(0, 0, 0);
        for (int i = 0; i < numHiddenLayerSize; i++)
        {
            currentOffset.x += offsetX;
             
            for (int j = 0; j < hiddenLayerSizeY.Length; j++)
            {

                currentOffset.y += offsetY;
                hiddenLayerSizeY[j] = Instantiate(prefab, position.position + currentOffset, Quaternion.identity, position);
                currentLayer.Add(hiddenLayerSizeY[j]);
                allMyHiddenLayers[i] = new hiddenLayer(currentLayer.ToArray());
            }
            currentLayer.Clear();
            currentOffset.y = 0;

        }
        return allMyHiddenLayers;

    }


    private void CreateConnections(ref GameObject[] startArray,ref GameObject[] endArray )
    {
        GameObject[] linesRenderer = new GameObject[startArray.Length * endArray.Length];



        for (int i = 0; i < linesRenderer.Length; i++)
        {
            linesRenderer[i] = new GameObject("line " + i);
            //not optimal
            linesRenderer[i].AddComponent<LineRenderer>();
            LineRenderer myLinesRenderer = linesRenderer[i].GetComponent<LineRenderer>();
            myLinesRenderer.startWidth = 0.05f;
            myLinesRenderer.endWidth = 0.03f;
            myLinesRenderer.material = new Material(Shader.Find("Sprites/Default"));
            myLinesRenderer.sortingOrder = 0;
            myLinesRenderer.startColor = Color.blue;
            myLinesRenderer.endColor = Color.yellow;
            myLinesRenderer.positionCount = 2;
            myLinesRenderer.useWorldSpace = true;

        }

        int countLines = 0;
        for (int i = 0; i < startArray.Length; i++)
        {

            for (int j = 0; j < endArray.Length; j++)
            {
                linesRenderer[countLines].transform.SetParent(startArray[i].transform);
                linesRenderer[countLines].GetComponent<LineRenderer>().SetPosition(0, startArray[i].transform.position);
                linesRenderer[countLines].GetComponent<LineRenderer>().SetPosition(1, startArray[i].transform.position);
                countLines++;
            }
        }
    }
    private void moveLines(ref GameObject[] startArray, ref GameObject[] endArray,int layerIndex)
    {
        Vector3 dir = Vector3.zero;
        LineRenderer myLinesPerLayer;
        int weightIndex=0;


        for (int i = 0; i < startArray.Length; i++)
        {

            int j = 0;
            foreach (Transform item in startArray[i].transform)
            {

                dir = (endArray[j].transform.position - startArray[i].transform.position).normalized;
                myLinesPerLayer = item.GetComponent<LineRenderer>();
              
                Color col;
                float [] normalizedWeights = NormilizeWeights(myNet.layers[layerIndex].weightsIn);
                float colorR =Mathf.Abs(normalizedWeights[weightIndex]);
                if (backPropTrain||gradientDescentTrain)
                {
                    if (normalizedWeights[i]>0)
                    {
                        col = new Color(0, colorR, 0, 1);
                        Color lerpedColor = Color.Lerp(myLinesPerLayer.endColor, col, onTrainTimerColorLerp);
                        myLinesPerLayer.startColor = Color.blue;
                        myLinesPerLayer.endColor = lerpedColor;
                    }
                    else
                    {
                        col = new Color(colorR, 0, 0, 1);
                        Color lerpedColor = Color.Lerp(myLinesPerLayer.endColor, col, onTrainTimerColorLerp);
                        myLinesPerLayer.startColor = Color.blue;
                        myLinesPerLayer.endColor = lerpedColor;
                    }


                }
                Vector3 myCurrentVector = myLinesPerLayer.GetPosition(1) - myLinesPerLayer.GetPosition(0);
                Vector3 newPos = myLinesPerLayer.GetPosition(1) + dir * Time.fixedDeltaTime * speedAnim;

                if (myLinesPerLayer.GetPosition(1).x > endArray[j].transform.position.x)
                {
                    myLinesPerLayer.SetPosition(1, endArray[j].transform.position);
                    j++;
                    continue;
                }
                myLinesPerLayer.SetPosition(1, newPos);
                j++;
                weightIndex++;


            }
            j = 0;
        }

    }


    /// <summary>
    /// Test data function
    /// </summary>
    /// <param name="startArray"></param>
    /// <param name="layerIndex"></param>
    /// <param name="dataInput"></param>
    void SetOutputCols(ref GameObject[] startArray, int layerIndex, double[] dataInput)
    {
    
            float[] inputsWeights;
            inputsWeights = NormilizeWeights(dataInput);
            inputsWeights[0] = (float)Math.Round(inputsWeights[0], 6);
            inputsWeights[1] = (float)Math.Round(inputsWeights[1], 6);
            inputsWeights[2] = (float)Math.Round(inputsWeights[2], 6);
             Color inputColor;
            for (int i = 0; i < inputs.Length; i++)
            {
                if (gradientDescentTrain || backPropTrain)
                {
                    inputColor = new Color(inputsWeights[0], inputsWeights[1], inputsWeights[2], 1);
                    Color lerpedColor = Color.Lerp(inputs[i].GetComponent<SpriteRenderer>().color, inputColor, onTrainTimerColorLerp);
                    inputs[i].GetComponent<SpriteRenderer>().color = lerpedColor;
                Color outputColor = new Color(inputsWeights[0], inputsWeights[1], inputsWeights[2], 1);
                SetTextColor(outputColor, ref inputCol);
            }
                else
                {
                    inputColor = new Color(inputsWeights[0], inputsWeights[1], inputsWeights[2], 1);
                    inputs[i].GetComponent<SpriteRenderer>().color = inputColor;
                Color outputColor = new Color(inputsWeights[0], inputsWeights[1], inputsWeights[2], 1);
                SetTextColor(outputColor, ref inputCol);
            }

            }

            for (int i = 0; i < startArray.Length; i++)
            {
                if (gradientDescentTrain || backPropTrain)
                {
                    inputsWeights = NormilizeWeights(myNet.layers[layerIndex].weightedInputs);
                    inputColor = new Color(0, inputsWeights[i], inputsWeights[i], 1);
                    Color lerpedColor = Color.Lerp(startArray[i].GetComponent<SpriteRenderer>().color, inputColor, onTrainTimerColorLerp);
                    startArray[i].GetComponent<SpriteRenderer>().color = lerpedColor;
                }
                else
                {
                    inputsWeights = NormilizeWeights(myNet.layers[layerIndex].weightedInputs);
                    inputColor = new Color(inputsWeights[i], inputsWeights[i], inputsWeights[i], 1);
                    startArray[i].GetComponent<SpriteRenderer>().color = inputColor;
                }   

            }

            inputsWeights = NormilizeWeights(myNet.layers[myNet.layers.Length - 1].weightedInputs);
            inputsWeights[0] = (float)Math.Round(inputsWeights[0], 5);
            inputsWeights[1] = (float)Math.Round(inputsWeights[1], 5);
            inputsWeights[2] = (float)Math.Round(inputsWeights[2], 5);
        if (gradientDescentTrain || backPropTrain)
        {
            inputColor = new Color(inputsWeights[0], 0, 0, 6);
            Color lerpedColor = Color.Lerp(outputs[0].GetComponent<SpriteRenderer>().color, inputColor, onTrainTimerColorLerp);
            outputs[0].GetComponent<SpriteRenderer>().color = lerpedColor;
            inputColor = new Color(0, inputsWeights[1], 0, 6);
            lerpedColor = Color.Lerp(outputs[1].GetComponent<SpriteRenderer>().color, inputColor, onTrainTimerColorLerp);
            outputs[1].GetComponent<SpriteRenderer>().color = lerpedColor;
            inputColor = new Color(0, 0, inputsWeights[2], 6);
            lerpedColor = Color.Lerp(outputs[2].GetComponent<SpriteRenderer>().color, inputColor, onTrainTimerColorLerp);
            outputs[2].GetComponent<SpriteRenderer>().color = lerpedColor;
            Color outputColor = new Color(inputsWeights[0], inputsWeights[1], inputsWeights[2], 1);
            SetTextColor(outputColor, ref outputCol);


        }
        else
        {

            inputColor = new Color(inputsWeights[0], 0.0f, 0.0f, 1);
            outputs[0].GetComponent<SpriteRenderer>().color = inputColor;

            inputColor = new Color(0.0f,inputsWeights[1], 0.0f, 1);
            outputs[1].GetComponent<SpriteRenderer>().color = inputColor;

            inputColor = new Color(0.0f, 0.0f, inputsWeights[2], 1);
            outputs[2].GetComponent<SpriteRenderer>().color = inputColor;

            Color outputColor=new Color(inputsWeights[0], inputsWeights[1], inputsWeights[2],1);
            SetTextColor(outputColor, ref outputCol);

        }
          
            
        

    }

    void SetTextColor(Color color,ref TextMeshProUGUI text)
    {
        text.color = color;

    }
    #endregion
    float[] NormilizeWeights(double[] weights)
    {
        float[] normalizedWeights = new float[weights.Length];
        float highestWeight = 0;
        for (int i = 0; i < weights.Length; i++)
        {
            if (weights[i] > highestWeight)
            {
                highestWeight = (float)weights[i];
            }

            normalizedWeights[i] = (float)weights[i];

        }
        for (int i = 0; i < weights.Length; i++)
        {
            normalizedWeights[i] = normalizedWeights[i] / highestWeight;
        }

        return normalizedWeights;

    }


    public void RecreateNet()
    {
        foreach (Transform child in spawnPointsInputs)
        {
            Destroy(child.gameObject);
        }
        foreach (Transform child in spawnPointsOutputs)
        {
            Destroy(child.gameObject);
        }
        foreach (Transform child in spawnPointsHiddenLayerSizeY)
        {
            Destroy(child.gameObject);
        }
        CreateBackendNet();
        CreateNetUI(ref numInputs, ref numHiddenLayers, ref numHiddenLayerSize, ref numOutputs);
    }

    public void OnValuesChange()
    {

        if (lastNumInputs != numInputs) RecreateNet();
        if (lastNumOutputs != numOutputs) RecreateNet();
        if (lastNumHiddenLayers != numHiddenLayers) RecreateNet();
        if (lastNumHiddenLayerSize != numHiddenLayerSize)RecreateNet();

        lastNumHiddenLayers = numHiddenLayers;
        lastNumHiddenLayerSize = numHiddenLayerSize;
        lastNumInputs = numInputs;
        lastNumOutputs = numOutputs;

    }

    public void SetActivationFunction(System.Int32 activationIndex)
    {
        myNet.SetNetActivations((ActivationType)activationIndex);
    }


    [System.Serializable]
    public struct hiddenLayer
    {
        public GameObject[] hiddenLayerSizeY;
        public hiddenLayer(GameObject[] hiddenLayerSizeY)
        {
           this.hiddenLayerSizeY = hiddenLayerSizeY; 
        }
    }
    public struct Batch {
        public data [] data;
        public Batch(data[] data)
        {
            this.data = data;
        }
    }

}
