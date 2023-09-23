using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public static class Activations 
{
  
    public static double Activation(double x,ActivationType activatedFunc)
    {
        switch (activatedFunc)
        {
            case ActivationType.Sigmoid:
                return Sigmoid(x);
            case ActivationType.Tanh:
                return Tanh(x);
            case ActivationType.ReLU:
                return ReLU(x);
            case ActivationType.LeakyReLU:
                return LeakyReLU(x);
            case ActivationType.Softmax:
                return Softmax(x);
            default:
                return Tanh(x);
        }
    }   

    public static double ActivationDerivative(double x, ActivationType activatedFunc) {
        switch (activatedFunc)
        {
            case ActivationType.Sigmoid:
                return SigmoidDerivative(x);
            case ActivationType.Tanh:
                return TanhDerivative(x);
            case ActivationType.ReLU:
                return ReLUDerivative(x);
            case ActivationType.LeakyReLU:
                return LeakyReLUDerivative(x);
            case ActivationType.Softmax:
                return SoftmaxDerivative(x);
            default:
                return TanhDerivative(x);
        }

    }








    //Activation Functions----------------------------------
    public static double Sigmoid(double x)
    {
        return 1 / (1 + System.Math.Exp(-x));
    }
    public static double Tanh(double x)
    {
        return System.Math.Tanh(x);
    }
    public static double ReLU(double x)
    {
        return System.Math.Max(0, x);
    }
    public static double LeakyReLU(double x)
    {
        return System.Math.Max(0.01 * x, x);
    }
    public static double Softmax(double x)
    {
        return System.Math.Exp(x);
    }

    //Derivatives------------------------------------------

    public static double SigmoidDerivative(double x)
    {
        return Sigmoid(x) * (1 - Sigmoid(x));
    }
    public static double TanhDerivative(double x)
    {
        return 1 - System.Math.Pow(Tanh(x), 2);
    }
    public static double SoftmaxDerivative(double x)
    {
        return Softmax(x) * (1 - Softmax(x));
    }
    public static double ReLUDerivative(double x)
    {
        if (x > 0)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }
    public static double LeakyReLUDerivative(double x)
    {
        if (x > 0)
        {
            return 1;
        }
        else
        {
            return 0.01;
        }
    }

}

public enum ActivationType
{
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    Softmax
}