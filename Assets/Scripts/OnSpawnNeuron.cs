using System.Collections;
using System.Collections.Generic;
using System.Threading;
using Unity.Mathematics;
using UnityEngine;

public class OnSpawnNeuron : MonoBehaviour
{
    public float timerSpawn;
    public float animationTime;
    float scaler=0.61f;

    void Start()
    {


    }
    private void OnEnable()
    {
    }
    void Update()
    {
        //if (scaler <= 0.6f)
        //{
        //    return;
        //}
        if (timerSpawn <= animationTime)
        {
            timerSpawn += Time.deltaTime;
            OnSpawn();
            Vector3 scalerVec = new Vector3(scaler, scaler, scaler);
            transform.localScale = scalerVec;
        }


    }
    public void OnSpawn()
    {
        //(Mathf.Pow(-timerSpawn, 2) + (4 * timerSpawn) + 2) * 0.01f
        float timerInAngle = Mathf.Rad2Deg * timerSpawn;
       scaler = Mathf.Lerp(0.334f, 1f, Mathf.Sin(Mathf.Pow(-timerInAngle*0.01f, 2)+(3* timerInAngle*0.01f) +0.8f));
       
    }
    
}
