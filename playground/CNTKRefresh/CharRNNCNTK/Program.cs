using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Runtime.Remoting.Messaging;
using System.Text;
using System.Threading.Tasks;
using CNTK;

namespace CharRNNCNTK
{
    class Program
    {
        //  Change the Device that will be used for training and evaluation
        //  If null is passed, it will choose whatever device is suitable
        private DeviceDescriptor device;

        void SetDevice(DeviceDescriptor device = null)
        {
            if (device == null) this.device = DeviceDescriptor.UseDefaultDevice();
            else this.device = device;
        }

        //  Data required for the prediction
        private string text;
        private List<char> characters;
        private Dictionary<char, int> charToIndex;
        private Dictionary<int, char> indexToChar;

        bool loadData(string filename)
        {
            if (!File.Exists(filename)) return false;
            text = File.ReadAllText(filename);
            characters = text.Distinct().ToList();
            characters.Sort();

            charToIndex = new Dictionary<char, int>();
            indexToChar = new Dictionary<int, char>();
            for (int i = 0; i < characters.Count; i++)
            {
                var c = characters[i];
                charToIndex.Add(c, i);
                indexToChar.Add(i, c);
            }
            return true;
        }

        Func<Variable, Function> CreateModel(int numOutputDimension, int numLstmLayer, int numHiddenDimension)
        {
            return (input) =>
            {
                Function model = input;
                for (int i = 0; i < numLstmLayer; i++)
                {
                    model = Stabilizer.Build(model, device);
                    model = LSTM.Build(model, numHiddenDimension, device);
                }
                model = Dense.Build(model, numOutputDimension, device);
                return model;
            };
        }

        internal struct InputModel
        {
            public Variable InputSequence;
            public Variable LabelSequence;
        }

        InputModel CreateInputs(int vocabularyDimension)
        {
            var axis = new Axis("inputAxis");
            var inputSequence = Variable.InputVariable(new int[] { vocabularyDimension }, DataType.Float, "features", new List<Axis> { axis, Axis.DefaultBatchAxis() });
            var labels = Variable.InputVariable(new int[] { vocabularyDimension }, DataType.Float, "labels", new List<Axis> { axis, Axis.DefaultBatchAxis() });

            var inputModel = new InputModel
            {
                InputSequence = inputSequence,
                LabelSequence = labels
            };
            return inputModel;
        }

        internal struct MinibatchData
        {
            public List<float> InputSequence;
            public List<float> OutputSequence;
        }

        MinibatchData GetData(int index, int minibatchSize, string data, Dictionary<char, int> charToIndex, int vocabDimension)
        {
            var inputString = data.Substring(index, minibatchSize);
            var outputString = data.Substring(index + 1, minibatchSize);

            //  Handle EOF
            if (outputString.Length < minibatchSize)
                minibatchSize = outputString.Length;
            inputString = data.Substring(0, minibatchSize);

            List<float> inputSequence = new List<float>();
            List<float> outputSequence = new List<float>();

            for (int i = 0; i < inputString.Length; i++)
            {
                var inputCharacterIndex = charToIndex[inputString[i]];
                var inputCharOneHot = new float[vocabDimension];
                inputCharOneHot[inputCharacterIndex] = 1;
                inputSequence.AddRange(inputCharOneHot);

                var outputCharacterIndex = charToIndex[outputString[i]];
                var outputCharOneHot = new float[vocabDimension];
                outputCharOneHot[outputCharacterIndex] = 1;
                outputSequence.AddRange(outputCharOneHot);
            }

            return new MinibatchData
            {
                InputSequence = inputSequence,
                OutputSequence = outputSequence
            };
        }

        public Program()
        {
            SetDevice();
            var trainingFile = "tinyshakespeare.txt";
            if(!loadData(trainingFile)) return;
            Console.WriteLine($"Data { trainingFile } has { text.Length } characters, with { characters.Count } unique characters.");

            var inputModel = CreateInputs(characters.Count);
            var modelSequence = CreateModel(characters.Count, 2, 256);
            var model = modelSequence(inputModel.InputSequence);
            
            //  Setup the criteria (loss and metric)
            var crossEntropy = CNTKLib.CrossEntropyWithSoftmax(model, inputModel.LabelSequence);
            var errors = CNTKLib.ClassificationError(model, inputModel.LabelSequence);

            //  Instantiate the trainer object to drive the model training
            var learningRatePerSample = new TrainingParameterScheduleDouble(0.001);
            var momentumTimeConstant = CNTKLib.MomentumAsTimeConstantSchedule(1100);
            var additionalParameters = new AdditionalLearningOptions
            {
                gradientClippingThresholdPerSample = 5.0,
                gradientClippingWithTruncation = true
            };
            var learner = Learner.MomentumSGDLearner(model.Parameters(), learningRatePerSample, momentumTimeConstant, false, additionalParameters);
            var trainer = Trainer.CreateTrainer(model, crossEntropy, errors, new List<Learner>(){ learner });

            var epochs = 50;
            var minibatchSize = 100;
            var maxNumberOfMinibatches = int.MaxValue;
            var sampleFrequency = 1000;
            var minibatchesPerEpoch = Math.Min(text.Length / minibatchSize, maxNumberOfMinibatches / epochs);
            var parameterTensor = model.Parameters();
            var sumOfParameters = 0;
            foreach (var parameter in parameterTensor)
            {
                sumOfParameters += parameter.Shape.TotalSize;
            }
            Console.WriteLine($"Training { sumOfParameters } parameter in { parameterTensor.Count } parameter tensors");
            Console.WriteLine($"Running { epochs } epochs with { minibatchesPerEpoch } minibatches per epoch");
            Console.WriteLine();

            for (int i = 0; i < epochs; i++)
            {
                var start = DateTime.Now;
                Console.WriteLine($"Running training on epoch {i + 1} of {epochs}");
                for (int j = 0; j < minibatchesPerEpoch; j++)
                {
                    var trainingData = GetData(j, minibatchSize, text, charToIndex, characters.Count);
                    var arguments = new Dictionary<Variable, Value>();
                    var features = Value.CreateSequence<float>(inputModel.InputSequence.Shape,
                        trainingData.InputSequence, device);
                    arguments.Add(inputModel.InputSequence, features);
                    var labels = Value.CreateSequence(inputModel.LabelSequence.Shape,
                        trainingData.OutputSequence, device);
                    arguments.Add(inputModel.LabelSequence, labels);
                    trainer.TrainMinibatch(arguments, device);

                    var globalMinibatch = i * minibatchesPerEpoch + j;
                    if (globalMinibatch % sampleFrequency == 0)
                        Sample(model, 50);
                    if (globalMinibatch % 100 == 0)
                    {
                        var minibatchId = j + 1;
                        var minibatchEndId = j + 100;
                        var trainingLossValue = trainer.PreviousMinibatchLossAverage();
                        var evaluationValue = trainer.PreviousMinibatchEvaluationAverage();
                        Console.WriteLine(
                            $"Epoch {(i + 1), 3}: Minibatch [{minibatchId, 6}-{minibatchEndId, 6}] CrossEntropyLoss = {trainingLossValue:F6}, EvaluationCriterion = {evaluationValue:F3}");
                    }
                }
                var end = DateTime.Now;
                var epochLength = end - start;
                Console.WriteLine(
                    $"Finished epoch {i + 1} in {epochLength.TotalSeconds} seconds ({epochLength.Hours:00}:{epochLength.Minutes:00}:{epochLength.Seconds:00}.{epochLength.Milliseconds:000})");
                var modelFilename = $"newmodels/shakespeare_epoch{ i + 1 }.dnn";
                model.Save(modelFilename);
                Console.WriteLine($"Saved model to { modelFilename }");
            }
            Console.ReadKey();
        }

        static void Main(string[] args)
        {
            new Program();
        }

        private void Sample(Function model, int targetLength)
        {
            var numberOfEvaluatedCharacters = 0;
            var textToTest = "";
            List<int> textOutput = new List<int>();

            //  Put the inputVariable, outputVariable, inputs, and outputs variable outside
            //  of any loops, as they may be accessed across the loops.
            Variable inputVariable = model.Arguments.Single();
            Variable outputVariable = model.Output;

            Dictionary<Variable, Value> inputs;
            Dictionary<Variable, Value> outputs = null;

            //  This list contain all of the inputs. It's size is determined by
            //  the number of unique characters * number of input.
            List<float> sequence = new List<float>();

            if (string.IsNullOrEmpty(textToTest))
                textToTest += indexToChar[new Random().Next(characters.Count)];

            for (int i = 0; i < textToTest.Length; i++)
            {
                var c = textToTest[i];
                var cAsIndex = charToIndex[c];
                textOutput.Add(cAsIndex);

                var input = new float[characters.Count];
                input[cAsIndex] = 1;
                sequence.AddRange(input);

                Value inputValue = Value.CreateSequence<float>(inputVariable.Shape, sequence, device);
                inputs = new Dictionary<Variable, Value>() { { inputVariable, inputValue } };
                outputs = new Dictionary<Variable, Value>() { { outputVariable, null } };
                model.Evaluate(inputs, outputs, device);
                numberOfEvaluatedCharacters++;
            }

            for (int i = 0; i < targetLength - textToTest.Length; i++)
            {
                var outputData = outputs[outputVariable].GetDenseData<float>(outputVariable);
                var suggestedCharIndex = GetSuggestion(outputData[0].ToList(), numberOfEvaluatedCharacters);
                textOutput.Add(suggestedCharIndex);

                var input = new float[characters.Count];
                input[suggestedCharIndex] = 1;
                sequence.AddRange(input);

                Value inputValue = Value.CreateSequence<float>(inputVariable.Shape, sequence, device);
                inputs = new Dictionary<Variable, Value>() { { inputVariable, inputValue } };
                outputs = new Dictionary<Variable, Value>() { { outputVariable, null } };
                model.Evaluate(inputs, outputs, device);
                numberOfEvaluatedCharacters++;
            }

            List<char> sentenceAsChar = textOutput.Select(x => indexToChar[x]).ToList();
            string sentence = string.Join("", sentenceAsChar.ToArray());
            Console.WriteLine(sentence);
        }

        int GetSuggestion(List<float> probabilities, int numberOfEvaluatedCharacters)
        {
            probabilities = probabilities.GetRange(characters.Count * (numberOfEvaluatedCharacters - 1), characters.Count);
            probabilities = probabilities.Select(f => (float)Math.Exp(f)).ToList();
            var sumOfProbabilities = probabilities.Sum();
            probabilities = probabilities.Select(x => x / sumOfProbabilities).ToList();
            var selection = 0;
            var randomValue = (float)new Random().NextDouble();
            foreach (var probability in probabilities)
            {
                randomValue -= probability;
                if (randomValue < 0) break;
                selection++;
            }
            return selection;
        }
    }
}
