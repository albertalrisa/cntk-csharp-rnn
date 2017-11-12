using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace CharRNNCNTK
{
    /// <summary>
    /// This class builds an embedding layer and returns it as a Function
    /// 
    /// Adapted from CNTK C# Training Example for LSTM Sequence Classifier
    /// https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/LSTMSequenceClassifier.cs
    /// </summary>
    class Embedding
    {
        /// <summary>
        /// This function builds an embedding layer from an input variable and returns it as a Function
        /// that can be passed through another layers.
        /// </summary>
        /// <typeparam name="T">The data type of the values. May be set to float or double.</typeparam>
        /// <param name="input">The input of the embedding layer</param>
        /// <param name="embeddingDimension">The number of dimensions in the embedding layer</param>
        /// <param name="device">Device used for the computation of this layer</param>
        /// <param name="outputName">The name of the Function instance in the network</param>
        /// <returns>A function that implements the embedding layer</returns>
        public static Function Build<T>(Function input, int embeddingDimension, DeviceDescriptor device, string outputName = "embedding")
        {
            System.Diagnostics.Debug.Assert(typeof(T) == typeof(float) || typeof(T) == typeof(double));
            System.Diagnostics.Debug.Assert(input.Arguments[0].Shape.Rank == 1);
            int inputDimension = input.Arguments[0].Shape[0];
            var targetType = typeof(T) == typeof(float) ? DataType.Float : DataType.Double;
            var embeddingParameters = new Parameter(new int[] { embeddingDimension, inputDimension }, targetType, CNTKLib.GlorotUniformInitializer(), device);
            return CNTKLib.Times(embeddingParameters, input, name:outputName);
        }

        public static Function Build(Function input, int embeddingDimension, DeviceDescriptor device, string outputName = "embedding")
        {
            return Build<float>(input, embeddingDimension, device, outputName);
        }
    }
}
