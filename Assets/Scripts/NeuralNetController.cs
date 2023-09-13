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

    public GameObject inputPrefab;
    public GameObject hiddenLayerPrefab;
    public GameObject outputPrefab;


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
    public bool dynamicNet = false;
    public bool trainNet = false;

    [Header("Debug")]
    public Transform spawnPointsInputs;
    public Transform spawnPointsHiddenLayersX;
    public Transform spawnPointsHiddenLayerSizeY;
    public Transform spawnPointsOutputs;
    public TextMeshProUGUI cost;


    private hiddenLayer[] allMyHiddenLayers;
    [SerializeField]
    private float speedAnim;
    [SerializeField]
    public NeuralNet myNet;
    data myData;
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
        if (trainNet)
        {
            myNet.Train(myData);
            cost.text = myNet.CostCalculation(myData).ToString();
        }
        //lines UI
        moveLines(ref inputs, ref allMyHiddenLayers[0].hiddenLayerSizeY,0);
        for (int i = 0; i < allMyHiddenLayers.Length; i++)
        {
            if (i + 1 == allMyHiddenLayers.Length)
            {
                break;
            }
            moveLines(ref allMyHiddenLayers[i].hiddenLayerSizeY, ref allMyHiddenLayers[i + 1].hiddenLayerSizeY,i);

        }
        moveLines(ref allMyHiddenLayers[allMyHiddenLayers.Length - 1].hiddenLayerSizeY, ref outputs, allMyHiddenLayers.Length);
    }

    private void CreateBackendNet()
    {
        //backend

        //dataset
        double[] outputs = new double[numOutputs];
        double[] inputs = new double[numInputs];

        inputs[0] = 1;
        inputs[1] = 0;
        inputs[2] = 0;

        outputs[0] = 1;
        outputs[1] = 0;


        myNet = new NeuralNet(numInputs, numHiddenLayers, numHiddenLayerSize, numOutputs);
        myNet.ConstructNet();
        myData = new data(inputs, outputs);
        myNet.SetDataset(myData);
        myNet.FillWeights();
        myNet.CalculateNet();

    }

    #region screen
    public void Train()
    {
        trainNet = !trainNet;
    }
    public void CalculateCost()
    {
        
        cost.text = myNet.CostCalculation(myData).ToString();
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
        LineRenderer[] myLinesPerLayer = new LineRenderer[startArray.Length * endArray.Length];
        int counter = 0;
        for (int i = 0; i < startArray.Length; i++)
        {
            int j = 0;
            foreach (Transform item in startArray[i].transform)
            {

                dir = (endArray[j].transform.position - startArray[i].transform.position).normalized;
                myLinesPerLayer[counter] = item.GetComponent<LineRenderer>();
                float colorR =(float) myNet.layers[layerIndex].weightsIn[counter];
                Color col = new Color(0, colorR, 0, 1);
                myLinesPerLayer[counter].startColor = col;
                myLinesPerLayer[counter].endColor = col;
                Vector3 myCurrentVector = myLinesPerLayer[counter].GetPosition(1) - myLinesPerLayer[counter].GetPosition(0);
                Vector3 newPos = myLinesPerLayer[counter].GetPosition(1) + dir * Time.deltaTime * speedAnim;

                if (myLinesPerLayer[counter].GetPosition(1).x > endArray[j].transform.position.x)
                {
                    myLinesPerLayer[counter].SetPosition(1, endArray[j].transform.position);
                    j++;
                    continue;
                }
                myLinesPerLayer[counter].SetPosition(1, newPos);
                j++;

            }
            j = 0;
        }

    }
    #endregion



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

        CreateNetUI(ref numInputs, ref numHiddenLayers, ref numHiddenLayerSize, ref numOutputs);
        CreateBackendNet();
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



    [System.Serializable]
    public struct hiddenLayer
    {
        public GameObject[] hiddenLayerSizeY;
        public hiddenLayer(GameObject[] hiddenLayerSizeY)
        {
           this.hiddenLayerSizeY = hiddenLayerSizeY; 
        }
    }
}
