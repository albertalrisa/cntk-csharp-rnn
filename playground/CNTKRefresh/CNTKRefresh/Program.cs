using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CNTK;

//  Character Language Model
//  CNTK Example
//  Ported from https://github.com/Microsoft/CNTK/blob/master/Examples/Text/CharacterLM/char_rnn.py

namespace CNTKRefresh
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
        private List<char> characters;
        private Dictionary<char, int> charToIndex;
        private Dictionary<int, char> indexToChar;

        bool loadData(string filename)
        {
            if (!File.Exists(filename)) return false;
            var corpus = File.ReadAllText(filename);
            characters = corpus.Distinct().ToList();
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

        //  Load and store the pre-trained model
        private Function model;

        bool loadModel(string filename)
        {
            if (!File.Exists(filename)) return false;
            model = Function.Load(filename, device);
            return true;
        }

        //  Function to pick random suggestion based on the probabilities generated
        //  by the neural network
        //  The formula is actually just an exponent (to make sure that all values are positive, 
        //  with smaller value represented as a small decimal value and vice versa. Then, the
        //  value are normalized and used for probability with (as the sum is now 1).
        int GetSuggestion(List<float> probabilities)
        {
            probabilities = probabilities.GetRange(characters.Count * (numberOfEvaluatedCharacters - 1), characters.Count);
            probabilities = probabilities.Select(f => (float)Math.Exp(f)).ToList();
            var sumOfProbabilities = probabilities.Sum();
            probabilities = probabilities.Select(x => x / sumOfProbabilities).ToList();
            var selection = 0;
            var randomValue = (float) new Random().NextDouble();
            foreach (var probability in probabilities)
            {
                randomValue -= probability;
                if (randomValue < 0) break;
                selection++;
            }
            return selection;
        }

        //  Variable to store how many characters has been evaluated. 
        //  This is used specifically to determine which part of the sequence should
        //  be read in the GetSuggestion function
        private int numberOfEvaluatedCharacters = 0;

        public Program()
        {
            SetDevice();
            if (!loadData("tinyshakespeare.txt")) return;
            if (!loadModel("shakespeare_epoch41.dnn")) return;

            var textToTest = "He";
            var targetLength = 200;
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
                var suggestedCharIndex = GetSuggestion(outputData[0].ToList());
                textOutput.Add(suggestedCharIndex);

                var input = new float[characters.Count];
                input[suggestedCharIndex] = 1;
                sequence.AddRange(input);

                Value inputValue = Value.CreateSequence<float>(inputVariable.Shape, sequence, device);
                inputs = new Dictionary<Variable, Value>(){ {inputVariable, inputValue } };
                outputs = new Dictionary<Variable, Value>() { { outputVariable, null } };
                model.Evaluate(inputs, outputs, device);
                numberOfEvaluatedCharacters++;
            }

            List<char> sentenceAsChar = textOutput.Select(x => indexToChar[x]).ToList();
            string sentence = string.Join("", sentenceAsChar.ToArray());
            Console.WriteLine(sentence);
            Console.ReadKey();
        }

        static void Main(string[] args)
        {
            new Program();
        }
    }
}
