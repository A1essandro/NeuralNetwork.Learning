using NeuralNetwork.Structure.Contract.Nodes;
using System.Diagnostics;

namespace NeuralNetwork.Learning.Strategies
{

    [DebuggerDisplay("Neuron:{Neuron}; Sigma:{Sigma};")]
    internal class NeuronSigma
    {

        public NeuronSigma(ISlaveNode neuron, double sigma)
        {
            Neuron = neuron;
            Sigma = sigma;
        }

        public ISlaveNode Neuron { get; }

        public double Sigma { get; }

    }


}