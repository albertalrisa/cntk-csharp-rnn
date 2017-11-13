using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace CharRNNCNTK
{
    /// <summary>
    /// This class builds a Dense layer and returns it as a Function
    /// 
    /// Adapted from CNTK C# Training Example Test Helper
    /// https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/TestHelper.cs
    /// </summary>
    class Dense
    {
        public enum Activation
        {
            None,
            ReLU,
            Sigmoid,
            Tanh
        }

        /// <summary>
        /// This function builds a Dense (fully connected linear) layer from an input variable and returns it as a Function
        /// that can be passed through another layers.
        /// </summary>
        /// <param name="input">The input of the Dense layer</param>
        /// <param name="outputDimension">The number of output nodes of the Dense layer</param>
        /// <param name="device">Device used for the computation of this layer</param>
        /// <param name="activation">The activation function used for this Dense layer</param>
        /// <param name="outputName">The name of the Function instance in the network</param>
        /// <returns>A function that implements the desired fully-connected layer.</returns>
        public static Function Build<T>(Variable input, int outputDimension, DeviceDescriptor device,
            Activation activation = Activation.None, string outputName = "Dense")
        {
            if (input.Shape.Rank != 1)
            {
                int newDimension = input.Shape.Dimensions.Aggregate((d1, d2) => d1 * d2);
                input = CNTKLib.Reshape(input, new int[] { newDimension });
            }

            Function fullyConnected = FullyConnectedLinearLayer<T>(input, outputDimension, device);
            Function dense;
            switch (activation)
            {
                default:
                case Activation.None:
                    dense = fullyConnected;
                    break;
                case Activation.ReLU:
                    dense = CNTKLib.ReLU(fullyConnected);
                    break;
                case Activation.Sigmoid:
                    dense = CNTKLib.Sigmoid(fullyConnected);
                    break;
                case Activation.Tanh:
                    dense = CNTKLib.Tanh(fullyConnected);
                    break;
            }
            return Function.Alias(dense, outputName);
        }

        public static Function Build(Variable input, int outputDimension, DeviceDescriptor device,
            Activation activation = Activation.None,
            string outputName = "Dense")
        {
            return Build<float>(input, outputDimension, device, activation, outputName);
        }

        private static Function FullyConnectedLinearLayer<T>(Variable input, int outputDimension, DeviceDescriptor device, string outputName = "")
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);
            int inputDimension = input.Shape[0];

            int[] s = { outputDimension, inputDimension };
            var weight = new Parameter((NDShape)s, DataType.Float,
                CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    1),
                device,
                "weight");
            var timesFunction = CNTKLib.Times(weight, input, "linearCombination");

            int[] s2 = { outputDimension };
            Parameter bias;
            if(typeof(T) == typeof(float))
                bias = new Parameter(s2, 0.0f, device, "bias");
            else
                bias = new Parameter(s2, 0.0, device, "bias");
            return CNTKLib.Plus(bias, timesFunction, outputName);
        }
    }
}
