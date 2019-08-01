using NeuralNetwork.Learning.Samples;
using NeuralNetwork.Structure.Contract.Networks;
using NeuralNetwork.Structure.Contract.Nodes;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace NeuralNetwork.Learning.Strategies
{
    public partial class BackpropagationStrategy : ILearningStrategy<ISimpleNetwork, ILearningSample>
    {

        public Task LearnSample(ISimpleNetwork network, ILearningSample sample, double theta)
        {
            return _teach(network, sample.Input, sample.Output, theta);
        }

        #region Private

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private async Task _teach(ISimpleNetwork network, IEnumerable<double> input, IEnumerable<double> expectation, double force)
        {
            await network.Input(input);
            var output = (await network.Output().ConfigureAwait(false)).ToArray();
            var sigmas = new List<NeuronSigma>();
            var expectationArr = expectation.ToArray();

            CalculateSigmasForOutputLayer(network, sigmas, force, output, expectationArr);

            foreach (var layer in network.Layers.Where(l => l != network.OutputLayer && l != network.InputLayer).Reverse())
            {
                foreach (var node in layer.Nodes.OfType<ISlaveNode>())
                {
                    var sigma = SigmaCalcForInnerLayers(network, sigmas, node);

                    sigmas.Add(new NeuronSigma(node, sigma));
                    ChangeWeights(network, node, sigma, force);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void CalculateSigmasForOutputLayer(ISimpleNetwork network, List<NeuronSigma> sigmas, double force, double[] output, double[] expectationArr)
        {
            var oIndex = 0;
            foreach (var node in network.OutputLayer.Nodes.OfType<ISlaveNode>())
            {
                var sigma = SigmaCalcForOutputLayer(expectationArr, node, output, oIndex);

                sigmas.Add(new NeuronSigma(node, sigma));
                ChangeWeights(network, node, sigma, force);
                oIndex++;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double SigmaCalcForInnerLayers(ISimpleNetwork network, IEnumerable<NeuronSigma> sigmas, ISlaveNode neuron)
        {
            var derivative = GetDerivative(neuron);
            return derivative * GetChildSigmas(network, sigmas, neuron);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double SigmaCalcForOutputLayer(IReadOnlyList<double> expectation, ISlaveNode neuron, IReadOnlyList<double> output, int oIndex)
        {
            var derivative = GetDerivative(neuron);
            return derivative * (expectation[oIndex] - output[oIndex]);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double GetDerivative(ISlaveNode neuron)
        {
            var x = neuron.Summator.LastCalculatedValue;
            return neuron.Function.GetDerivative(x);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ChangeWeights(ISimpleNetwork network, ISlaveNode neuron, double sigma, double force)
        {
            var synapses = network.Synapses.Where(x => x.SlaveNode == neuron);

            Parallel.ForEach(synapses, synapse =>
            {
                var masterNodeOutput = synapse.MasterNode.LastCalculatedValue;
                synapse.ChangeWeight(force * sigma * masterNodeOutput);
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static double GetChildSigmas(ISimpleNetwork network, IEnumerable<NeuronSigma> sigmas, INode neuron)
        {
            double sigma = 0;
            foreach (var neuronSigma in sigmas)
            {
                var synapses = network.Synapses.Where(x => x.SlaveNode == neuronSigma.Neuron && x.MasterNode == neuron);

                foreach (var synapse in synapses)
                {
                    sigma += synapse.Weight * neuronSigma.Sigma;
                }
            }
            return sigma;
        }

        #endregion

    }

}