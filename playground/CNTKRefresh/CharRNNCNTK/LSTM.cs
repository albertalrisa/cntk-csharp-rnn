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
    /// Adapted from CNTK C# Training Example for LSTM Sequence Classifier
    /// https://github.com/Microsoft/CNTK/blob/master/Examples/TrainingCSharp/Common/LSTMSequenceClassifier.cs
    /// Implemented according to the formula on Colah's Blog on Understanding LSTM
    /// http://colah.github.io/posts/2015-08-Understanding-LSTMs/
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
            return Function.AsComposite(lstmFunction, outputName);
            //return CNTKLib.SequenceLast(lstmFunction, outputName);
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
            int lstmOutputDimension = prevOutput.Shape[0];
            int lstmCellDimension = prevCellState.Shape[0];

            bool isFloatType = typeof(TElementType) == typeof(float);
            DataType dataType = isFloatType ? DataType.Float : DataType.Double;

            //  This is done according to the example in CNTK. The problem with this is that because it is called multiple times,
            //  the number of parameter tensor increases. Fix may be needed to make it identical to the one on Python.
            Func<int, Parameter> createBiasParameters;
            if(isFloatType) createBiasParameters = (dimension) => new Parameter(new [] {dimension}, 0.01f, device, "Bias");
            else createBiasParameters = (dimension) => new Parameter(new[] { dimension }, 0.01, device, "Bias");

            uint seed = 1;
            Func<int, Parameter> createWeightParameters = (outputDimension) => new Parameter(new [] { outputDimension, NDShape.InferredDimension }, 
                dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed++), device, "Weight");

            Function stabilizedPrevOutput;
            if (enableSelfStabilization)
                stabilizedPrevOutput = Stabilize.Build<TElementType>(prevOutput, device, "StabilizedPrevOutput");
            else
                stabilizedPrevOutput = prevOutput;

            Func<Variable> InputLinearCombinationPlusBias = () =>
                CNTKLib.Plus(createBiasParameters(lstmCellDimension),
                    (createWeightParameters(lstmCellDimension) * input), "LinearCombinationPlusBias");

            Func<Variable, Variable> PrevOutputLinearCombination = (previousOutput) =>
                CNTKLib.Times(createWeightParameters(lstmCellDimension), previousOutput);

            //  Forget Gate
            Function ft = 
                CNTKLib.Sigmoid(
                    InputLinearCombinationPlusBias() + PrevOutputLinearCombination(stabilizedPrevOutput),
                    "ForgetGate");

            //  Input Gate
            Function it =
                CNTKLib.Sigmoid(
                    InputLinearCombinationPlusBias() + PrevOutputLinearCombination(stabilizedPrevOutput),
                    "InputGate");
            Function ctt =
                CNTKLib.Tanh(
                    InputLinearCombinationPlusBias() + PrevOutputLinearCombination(stabilizedPrevOutput),
                    "CandidateValue");

            //  New Cell State
            Function ct =
                CNTKLib.Plus(CNTKLib.ElementTimes(ft, prevCellState), CNTKLib.ElementTimes(it, ctt));

            //  Output Gate
            Function ot =
                CNTKLib.Sigmoid(
                    InputLinearCombinationPlusBias() + PrevOutputLinearCombination(stabilizedPrevOutput),
                    "OutputGate");
            Function ht =
                CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct), "Output");
            Function stabilizedHt;
            if (enableSelfStabilization)
                stabilizedHt = Stabilize.Build<TElementType>(ht, device, "OutputStabilized");
            else
                stabilizedHt = ht;

            //  Prepare output
            Function c = ct;
            Function h = (lstmOutputDimension != lstmCellDimension)
                ? CNTKLib.Times(createWeightParameters(lstmOutputDimension), stabilizedHt, "Output")
                : stabilizedHt;

            return new Tuple<Function, Function>(h, c);
        }

//        LSTM with Peephole Implementation in CNTK Example
//        public static Tuple<Function, Function> LSTMPCell<ElementType>(
//            Variable input, Variable prevOutput, Variable prevCellState, bool enableSelfStabilization, DeviceDescriptor device)
//        {
//            int outputDimension = prevOutput.Shape[0];
//            int cellDimension = prevCellState.Shape[0];
//
//            bool isFloatType = typeof(ElementType) == typeof(float);
//            DataType dataType = isFloatType ? DataType.Float : DataType.Double;
//
//            Func<int, Parameter> createBiasParam;
//            if (isFloatType)
//                createBiasParam = (dimension) => new Parameter(new int[] { dimension }, 0.01f, device, "BiasParameter");
//            else
//                createBiasParam = (dimension) => new Parameter(new int[] { dimension }, 0.01, device, "BiasParameter");
//
//            uint seed2 = 1;
//            Func<int, Parameter> createProjectionParameter = (oDimension) => new Parameter(new int[] { oDimension, NDShape.InferredDimension },
//                dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device, "WeightMatrix");
//            Func<int, Parameter> createDiagWeightParameter = (dimension) => new Parameter(new int[] { dimension },
//                dataType, CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device, "StateWeightMatrix");
//
//            Function stabilizedPrevOutput;
//            Function stabilizedPrevCellState;
//            if (enableSelfStabilization)
//            {
//                stabilizedPrevOutput = Stabilize.Build<ElementType>(prevOutput, device, "StabilizedPrevOutput");
//                stabilizedPrevCellState = Stabilize.Build<ElementType>(prevCellState, device, "StabilizedPrevCellState");
//            }
//            else
//            {
//                stabilizedPrevOutput = prevOutput;
//                stabilizedPrevCellState = prevCellState;
//            }
//
//            Func<Variable> projectInput = () =>
//                CNTKLib.Plus(createBiasParam(cellDimension), (createProjectionParameter(cellDimension) * input), "LinearCombinationPlusBias");
//
//            //  Input gate
//            Function it =
//                CNTKLib.Sigmoid(
//                    (Variable)(projectInput() + (createProjectionParameter(cellDimension) * stabilizedPrevOutput)) +
//                    CNTKLib.ElementTimes(createDiagWeightParameter(cellDimension), stabilizedPrevCellState), "InputGate");
//            Function bit = CNTKLib.ElementTimes(
//                it,
//                CNTKLib.Tanh(projectInput() + (createProjectionParameter(cellDimension) * stabilizedPrevOutput)), "CandidateValue");
//
//            //  Forget-me-not gate
//            Function ft = CNTKLib.Sigmoid(
//                (Variable)(projectInput() + (createProjectionParameter(cellDimension) * stabilizedPrevOutput)) +
//                CNTKLib.ElementTimes(createDiagWeightParameter(cellDimension), stabilizedPrevCellState), "ForgetGate");
//            Function bft = CNTKLib.ElementTimes(ft, prevCellState, "forgetLastCellState");
//
//            //  Calculating new cell state
//            Function ct = CNTKLib.Plus(bft, bit, "CellState");
//            Function stabilizedCt;
//            if (enableSelfStabilization)
//                stabilizedCt = Stabilize.Build<ElementType>(ct, device, "CellStateStablized");
//            else
//                stabilizedCt = ct;
//
//            //  Output gate
//            Function ot = CNTKLib.Sigmoid(
//                (Variable)(projectInput() + (createProjectionParameter(cellDimension) * stabilizedPrevOutput)) +
//                CNTKLib.ElementTimes(createDiagWeightParameter(cellDimension), stabilizedCt), "OutputGate");
//            Function ht = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct), "Output");
//            Function stabilizedHt;
//            if (enableSelfStabilization)
//                stabilizedHt = Stabilize.Build<ElementType>(ht, device, "OutputStabilized");
//            else
//                stabilizedHt = ht;
//
//            //  Prepare output
//            Function c = ct;
//            Function h = (outputDimension != cellDimension)
//                ? CNTKLib.Times(createProjectionParameter(outputDimension), stabilizedHt, "Output")
//                : stabilizedHt;
//
//            return new Tuple<Function, Function>(h, c);
//        }
//
//        
//        private static Tuple<Function, Function> LSTMPComponent<ElementType>(
//            Variable input,
//            NDShape outputShape, NDShape cellShape,
//            Func<Variable, Function> recurrenceHookH,
//            Func<Variable, Function> recurrenceHookC,
//            bool enableSelfStabilization,
//            DeviceDescriptor device)
//        {
//            var dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
//            var dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);
//
//            var lstmCell = LSTMPCell<ElementType>(input, dh, dc, enableSelfStabilization, device);
//            var actualDh = recurrenceHookH(lstmCell.Item1);
//            var actualDc = recurrenceHookC(lstmCell.Item2);
//
//            (lstmCell.Item1).ReplacePlaceholders(new Dictionary<Variable, Variable> { { dh, actualDh }, { dc, actualDc } });
//            return new Tuple<Function, Function>(lstmCell.Item1, lstmCell.Item2);
//        }

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
