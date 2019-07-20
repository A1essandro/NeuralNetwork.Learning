using NeuralNetwork.Structure.Nodes;
using System.Diagnostics;

namespace NeuralNetwork.Learning.Strategies
{
    public partial class BackpropagationStrategy
    {

        [DebuggerDisplay("Neuron:{Neuron}; Sigma:{Sigma};")]
        private struct NeuronSigma
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

}