using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace CharRNNCNTK
{
    /// <summary>
    /// This class builds a Droppo self-stabilizer layer and outputs it as a function
    /// Based on https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/SelfLR.pdf
    /// <para>
    /// Adapted from CNTK C# Training Example for LSTM Sequence Classifier
    /// https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/LSTMSequenceClassifier.cs
    /// </para>
    /// </summary>
    class Stabilizer
    {
        /// <summary>
        /// This function builds an Stabilizer layer from an input variable that will multiply the input with a scalar 
        /// and returns it as a Function that can be passed through another layers.
        /// </summary>
        /// <typeparam name="TElementType">The data type of the values. May be set to float or double.</typeparam>
        /// <param name="input">The input of the Stabilizer layer</param>
        /// <param name="device">Device used for the computation of this layer</param>
        /// <param name="outputName">The name of the Function instance in the network</param>
        /// <returns>A Function that implements Stabilizer</returns>
        public static Function Build<TElementType>(Variable input, DeviceDescriptor device, string outputName = "Stabilizer")
        {
            System.Diagnostics.Debug.Assert(typeof(TElementType) == typeof(float) || typeof(TElementType) == typeof(double));
            bool isFloatType = typeof(TElementType) == typeof(float);
            Constant f, fInv;
            if (isFloatType)
            {
                f = Constant.Scalar(4.0f, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0f);
            }
            else
            {
                f = Constant.Scalar(4.0, device);
                fInv = Constant.Scalar(f.DataType, 1.0 / 4.0);
            }

            var beta = CNTKLib.ElementTimes(
                fInv,
                CNTKLib.Log(
                    Constant.Scalar(f.DataType, 1.0) +
                    CNTKLib.Exp(CNTKLib.ElementTimes(f,
                        new Parameter(new NDShape(), f.DataType, .99537863 /* 1/f*ln(e^f-1)*/, device, "alpha")))),
                "beta");
            return Function.Alias(CNTKLib.ElementTimes(beta, input), outputName);
        }

        public static Function Build(Variable input, DeviceDescriptor device, string outputName = "Stabilizer")
        {
            return Build<float>(input, device, outputName);
        }
    }
}
