using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using NeuroWnd.Neuron_definition;
using NeuroWnd.Activate_functions;

namespace NeuroWnd.Neuro_Nets
{
    public class NeuronLocation
    {
        public int Layer;
        public int Number;

        public NeuronLocation(int _layer, int _number)
        {
            Layer = _layer;
            Number = _number;
        }
    }

    public class NeuroNetLearningInterface
    {
        private NeuroNet learned_net;
        private string netName;
        private string selectionName;

        public int CountLayers { get { return learned_net.NeuronsInLayers.GetLength(0); } }
        public int CountNeurons { get { return learned_net.NeuronsCount; } }
        public int CountInputNeurons { get { return learned_net.InputNeuronsCount; } }
        public int CountOutputNeurons { get { return learned_net.OutputNeuronsCount; } }
        public bool IsIterationsFinished { get { return learned_net.IsIterationsFinished; } }
        public bool IsWaveCameToOutputNeuron { get { return learned_net.IsWaveCameToOutputNeuron; } }

        private Neuron getNeuron(NeuronLocation location)
        {
            return learned_net.GetNeuron(getNeuronIndexInNet(location));
        }
        private int getNeuronIndexInNet(NeuronLocation location)
        {
            if (location.Layer < 1 || location.Layer > learned_net.NeuronsInLayers.GetLength(0))
                throw new Exception("Invalid index of layer");

            int index = 0;
            for (int i = 0; i < location.Layer - 1; i++)
            {
                index += learned_net.NeuronsInLayers[i];
            }

            if (index + location.Number > learned_net.NeuronsCount ||
                index + location.Number < 1)
                throw new Exception("Invalid number of neuron in layer");

            return index + location.Number - 1;
        }
        private NeuronLocation getNeuronLocation(Neuron neuron)
        {
            int index = learned_net.GetIndexNeuron(neuron);
            if (index >= 0)
            {
                int layer = 1;
                int number = 1;

                int curInd = 0;
                for (int i = 0; i < learned_net.NeuronsInLayers.GetLength(0); i++)
                {
                    int nextInd = curInd + learned_net.NeuronsInLayers[i];
                    if (index < nextInd && index >= curInd)
                    {
                        number = index - curInd + 1;
                        break;
                    }
                    curInd = nextInd;
                    layer++;
                }

                return new NeuronLocation(layer, number);
            }
            else
            {
                return null;
            }
        }

        public NeuroNetLearningInterface(NeuroNet net, string _neuroNetName, string _selectionName)
        {
            learned_net = net;
            netName = _neuroNetName;
            selectionName = _selectionName;
        }

        public int GetCountNeuronsInLayer(int layerIndex)
        {
            if (layerIndex < 1 || layerIndex > learned_net.NeuronsInLayers.GetLength(0))
                throw new Exception("Invalid index of layer");

            return learned_net.NeuronsInLayers[layerIndex - 1];
        }
        public NeuronLocation[] GetInputsOfNeuron(NeuronLocation neuronLocation)
        {
            int indexCurNeuron = getNeuronIndexInNet(neuronLocation);
            Neuron neu = getNeuron(neuronLocation);
            NeuronLocation[] arr = new NeuronLocation[neu.InputsCount];

            int k = 0;
            for (int i = 0; i < learned_net.NeuronsCount; i++)
            {
                if (learned_net.ConnectionsOfNeurons[i, indexCurNeuron] == true)
                {
                    arr[k] = getNeuronLocation(learned_net.GetNeuron(i));
                    k++;
                }
            }

            return arr;
        }
        public NeuronLocation[] GetOutputsOfNeuron(NeuronLocation neuronLocation)
        {
            int indexCurNeuron = getNeuronIndexInNet(neuronLocation);
            Neuron neu = getNeuron(neuronLocation);
            
            int k = 0;
            for (int i = 0; i < learned_net.NeuronsCount; i++)
            {
                if (learned_net.ConnectionsOfNeurons[indexCurNeuron, i] == true)
                {
                    k++;
                }
            }

            NeuronLocation[] arr = new NeuronLocation[k];
            k = 0;
            for (int i = 0; i < learned_net.NeuronsCount; i++)
            {
                if (learned_net.ConnectionsOfNeurons[indexCurNeuron, i] == true)
                {
                    arr[k] = getNeuronLocation(learned_net.GetNeuron(i));
                }
            }

            return arr;
        }
        public bool IsConnection(NeuronLocation input, NeuronLocation output)
        {
            int indexInput = getNeuronIndexInNet(input);
            int indexOutput = getNeuronIndexInNet(output);

            return learned_net.ConnectionsOfNeurons[indexOutput, indexInput];
        }
        public double GetConnectionWeight(NeuronLocation input, NeuronLocation output)
        {
            int indexInput = getNeuronIndexInNet(input);
            int indexOutput = getNeuronIndexInNet(output);
            return learned_net.WeightsOfConnections[indexOutput, indexInput];
        }
        public Tuple<double, NeuronLocation>[] GetInputsOfNeuronWithWeights(NeuronLocation location)
        {
            NeuronLocation[] loc = GetInputsOfNeuron(location);
            Tuple<double, NeuronLocation>[] res = new Tuple<double, NeuronLocation>[loc.Length];
            for (int i = 0; i < loc.Length; i++)
            {
                res[i] = new Tuple<double, NeuronLocation>(GetConnectionWeight(loc[i], location), loc[i]);
            }
            return res;
        }
        public void ChangeConnectionWeight(NeuronLocation input, NeuronLocation output, double weight)
        {
            Neuron inp = getNeuron(input);
            Neuron oup = getNeuron(output);
            if (IsConnection(input, output) == true)
            {
                int indexOut = getNeuronIndexInNet(output);
                inp.SetWeightValue(weight, indexOut);
            }
            else
            {
                throw new Exception("Связь между нейронами не найдена");
            }
        }
        public void SetNewConnection(NeuronLocation input, NeuronLocation output, double weight)
        {
            Neuron inp = getNeuron(input);
            Neuron oup = getNeuron(output);
            if (IsConnection(input, output) == true)
            {
                throw new Exception("Связь между нейронами уже существует");
            }
            else
            {
                int indexIn = getNeuronIndexInNet(input);
                int indexOut = getNeuronIndexInNet(output);

                learned_net.SetNewConnection(indexIn, indexOut, weight);
            }
        }
        public void DeleteConnection(NeuronLocation input, NeuronLocation output)
        {
            int indexIn = getNeuronIndexInNet(input);
            int indexOut = getNeuronIndexInNet(output);

            learned_net.DeleteConnection(indexIn, indexOut);
        }
        public double GetOutputValueOfNeuron(NeuronLocation neuron)
        {
            return getNeuron(neuron).OutputValue;
        }

        //TODO:
        public string GetNameOfAF(NeuronLocation neuron)
        {
            return null;
        }
        public bool HasAFContinuousDerivative(NeuronLocation neuron)
        {
            return false;
        }
        public int GetCountParametersAF(NeuronLocation neuron)
        {
            return -1;
        }
        public string[] GetNamesOfParametersAF(NeuronLocation neuron)
        {
            return null;
        }
        public double GetValueOfParameterAF(NeuronLocation neuron, string parameterName)
        {
            return -1;
        }
        public void SetValueOfParameterAF(NeuronLocation neuron, string parameterName, double value)
        {
        }
        public double GetAFValue(NeuronLocation neuron, double x)
        {
            return -1;
        }
        public double GetDerivativeAFValue(NeuronLocation neuron, double x)
        {
            return -1;
        }
        public void SaveNeuroNetChanges(string learningAlgorithmName)
        {

        }

        public void ResetNeuroNet()
        {
            learned_net.ResetNeuroNet();
        }
        public double[] MakeStep(double[] inputs)
        {
            return learned_net.MakeStep(inputs);
        }
        public double[] MakeIteration(double[] inputs)
        {
            return learned_net.MakeIteration(inputs);
        }
        public double[] MakeAnswer(double[] inputs, double eps = 1E-16)
        {
            return learned_net.MakeAnswer(inputs, eps);
        }
    }
}
