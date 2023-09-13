using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.PlayerLoop;
using UnityEngine.UI;

[CustomEditor(typeof(NeuralNetController))]
public class NeuralNetEditor : Editor
{

    public override void OnInspectorGUI()
    {
        base.OnInspectorGUI();
        NeuralNetController myScript = (NeuralNetController)target;
    }
    
}
