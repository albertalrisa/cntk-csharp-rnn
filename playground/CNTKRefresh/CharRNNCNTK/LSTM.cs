using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace CharRNNCNTK
{
    /// <summary>
    /// This class builds an LSTM layer and returns it as a Function
    /// 
    /// Adapted from LSTM Block in Python implementation
    /// https://github.com/Microsoft/CNTK/blob/master/bindings/python/cntk/layers/blocks.py
    /// </summary>
    class LSTM
    {
        /// <summary>
        /// This function builds an LSTM layer from an input variable and returns it as a Function
        /// that can be passed through another layers.
        /// </summary>
        /// <typeparam name="T">The data type of the values. May be set to float or double.</typeparam>
        /// <param name="input">The input of the LSTM layer</param>
        /// <param name="lstmDimension">The output dimension of the LSTM layer</param>
        /// <param name="device">Device used for the computation of this layer</param>
        /// <param name="cellDimension">The cell shape of the LSTM. If left as 0, the shape of the cell will be equal to the lstmDimension.</param>
        /// <param name="enableSelfStabilization">If True, then all state-related projection will contain a Stabilizer()</param>
        /// <param name="outputName">The name of the Function instance in the network</param>
        /// <returns>A function that implements a recurrent LSTM layer</returns>
        public static Function Build<T>(Variable input,int lstmDimension, DeviceDescriptor device, int cellDimension = 0, bool enableSelfStabilization = false, string outputName = "lstm")
        {
            System.Diagnostics.Debug.Assert(typeof(T) == typeof(float) || typeof(T) == typeof(double));
            if (cellDimension == 0) cellDimension = lstmDimension;
            Func<Variable, Function> pastValueRecurrenceHook = (x) => CNTKLib.PastValue(x);
            Function lstmFunction = LSTMComponent<T>(input, new int[] {lstmDimension}, new int[] {cellDimension},
                    pastValueRecurrenceHook, pastValueRecurrenceHook, enableSelfStabilization, device)
                .Item1;
            return Function.Alias(lstmFunction, "LSTM");
        }

        public static Function Build(Variable input,int lstmDimension, DeviceDescriptor device,
            int cellDimension = 0, bool enableSelfStabilization = false, string outputName = "lstm")
        {
            return Build<float>(input, lstmDimension, device, cellDimension, enableSelfStabilization, outputName);
        }

        /// <summary>
        /// This function creates an LSTM block that implements one step of recurrence.
        /// It accepts the previous state and outputs its new state as a two-valued tuple (output, cell state)
        /// </summary>
        /// <typeparam name="TElementType">The data type of the values. May be set to float or double</typeparam>
        /// <param name="input">The input to the LSTM</param>
        /// <param name="prevOutput">The output of the previous step of the LSTM</param>
        /// <param name="prevCellState">The cell state of the previous step of the LSTM</param>
        /// <param name="enableSelfStabilization">If True, then all state-related projection will contain a Stabilizer()</param>
        /// <param name="device">Device used for the computation of this cell</param>
        /// <returns>A function (prev_h, prev_c, input) -> (h, c) that implements one step of a recurrent LSTM layer</returns>
        public static Tuple<Function, Function> LSTMCell<TElementType>(Variable input, Variable prevOutput,
            Variable prevCellState, bool enableSelfStabilization, DeviceDescriptor device)
        {
            //  TODO: Implements Self Stabilization
            //  TODO: Implements Peephole
            int lstmOutputDimension = prevOutput.Shape[0];
            int lstmCellDimension = prevCellState.Shape[0];

            bool isFloatType = typeof(TElementType) == typeof(float);
            DataType dataType = isFloatType ? DataType.Float : DataType.Double;

            if (enableSelfStabilization)
            {
                prevOutput = Stabilizer.Build(prevOutput, device, "StabilizedPrevOutput");
                prevCellState = Stabilizer.Build(prevCellState, device, "StabilizedPrevCellState");
            }

            uint seed = 1;
            Parameter W = new Parameter((NDShape) new[] { lstmCellDimension * 4, NDShape.InferredDimension }, dataType, 
                CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device, "W");
            Parameter b = new Parameter((NDShape) new[] { lstmCellDimension * 4 }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device, "b");

            Variable linearCombination = CNTKLib.Times(W, input, "linearCombinationInput");
            Variable linearCombinationPlusBias = CNTKLib.Plus(b, linearCombination, "linearCombinationInputPlusBias");

            Parameter H = new Parameter((NDShape) new[] { lstmCellDimension * 4, lstmOutputDimension }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++));
            Variable linearCombinationPrevOutput = CNTKLib.Times(H, prevOutput, "linearCombinationPreviousOutput");

            Variable gateInput = CNTKLib.Plus(linearCombinationPlusBias, linearCombinationPrevOutput, "gateInput");
            Variable forgetProjection = 
                CNTKLib.Slice(gateInput, new AxisVector() { new Axis(0) }, new IntVector() { lstmCellDimension * 0 }, new IntVector() { lstmCellDimension * 1 });
            Variable inputProjection =
                CNTKLib.Slice(gateInput, new AxisVector() { new Axis(0) }, new IntVector() { lstmCellDimension * 1 }, new IntVector() { lstmCellDimension * 2 });
            Variable outputProjection =
                CNTKLib.Slice(gateInput, new AxisVector() { new Axis(0) }, new IntVector() { lstmCellDimension * 2 }, new IntVector() { lstmCellDimension * 3 });
            Variable candidateProjection =
                CNTKLib.Slice(gateInput, new AxisVector() { new Axis(0) }, new IntVector() { lstmCellDimension * 3 }, new IntVector() { lstmCellDimension * 4 });

            Function ft = CNTKLib.Sigmoid(forgetProjection, "ForgetGate");
            Function it = CNTKLib.Sigmoid(inputProjection, "InputGate");
            Function ot = CNTKLib.Sigmoid(outputProjection, "OutputGate");
            Function ctt = CNTKLib.Tanh(candidateProjection, "Candidate");

            Function bft = CNTKLib.ElementTimes(prevCellState, ft);
            Function bit = CNTKLib.ElementTimes(it, ctt);
            Function ct = CNTKLib.Plus(bft, bit, "CellState");

            //  According to the TrainingCSharp example in CNTK repository, h (output) should be stabilized,
            //  however, the Python binding only stabilizes the previous output and previous cell state
            Function h = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct), "Output");
            Function c = ct;
            if (lstmOutputDimension != lstmCellDimension)
            {
                Parameter P = new Parameter((NDShape) new[] { lstmOutputDimension, lstmCellDimension }, dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++));
                h = CNTKLib.Times(P, h, "StandarizedOutput");
            }
            
            return new Tuple<Function, Function>(h, c);
        }

        private static Tuple<Function, Function> LSTMComponent<TElementType>(Variable input, NDShape outputShape,
            NDShape cellShape, Func<Variable, Function> recurrenceHookH, Func<Variable, Function> recurrenceHookC,
            bool enableSelfStabilization, DeviceDescriptor device)
        {
            var dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
            var dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);

            var lstmCell = LSTMCell<TElementType>(input, dh, dc, enableSelfStabilization, device);
            var actualDh = recurrenceHookH(lstmCell.Item1);
            var actualDc = recurrenceHookC(lstmCell.Item2);

            (lstmCell.Item1).ReplacePlaceholders(new Dictionary<Variable, Variable> {{dh, actualDh}, {dc, actualDc}});
            return new Tuple<Function, Function>(lstmCell.Item1, lstmCell.Item2);
        }
    }
}
