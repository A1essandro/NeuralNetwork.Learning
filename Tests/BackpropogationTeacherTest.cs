using NeuralNetwork.Learning;
using NeuralNetwork.Learning.Samples;
using NeuralNetwork.Learning.Strategies;
using NeuralNetwork.Structure.ActivationFunctions;
using NeuralNetwork.Structure.Contract.Layers;
using NeuralNetwork.Structure.Contract.Networks;
using NeuralNetwork.Structure.Contract.Nodes;
using NeuralNetwork.Structure.Layers;
using NeuralNetwork.Structure.Networks;
using NeuralNetwork.Structure.Nodes;
using NeuralNetwork.Structure.Synapses;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace Tests
{
    public class BackpropogationTeacherTest
    {

        private const double DELTA = 0.15;
        private const double THETA = 0.33;

        [Fact]
        public async Task TestTeachXor()
        {
            IInputLayer inputLayer = new InputLayer(() => new InputNode(), 2, new Bias());
            var innerLayer = new Layer(() => new Neuron(new Logistic(0.888)), 3, new Bias());
            var outputLayer = new Layer(new Neuron(new Logistic(0.777)));

            var network = new Network
            {
                InputLayer = inputLayer,
                OutputLayer = outputLayer
            };
            network.AddInnerLayer(innerLayer);

            foreach (var layer in network.Layers)
            {
                foreach(var node in layer.Nodes)
                {
                    node.OnResultCalculated += (n, v) =>
                    {
                        Debug.WriteLine($"{n}: {v}");

                        return Task.CompletedTask;
                    };
                }
            }

            var generator = new EachToEachSynapseGenerator(new Random());
            generator.Generate(network, inputLayer, innerLayer);
            generator.Generate(network, innerLayer, outputLayer);

            var samples = new List<ILearningSample>
            {
                new LearningSample(new double[] { 0, 1 }, new double[] { 1 }),
                new LearningSample(new double[] { 1, 0 }, new double[] { 1 }),
                new LearningSample(new double[] { 0, 0 }, new double[] { 0 }),
                new LearningSample(new double[] { 1, 1 }, new double[] { 0 })
            };

            await network.Input(new double[] { 1, 0 });

            var beforeLearning = network.LastCalculatedValue.First();

            var strategy = new BackpropagationStrategy();
            var settings = new LearningSettings
            {
                EpochRepeats = 10000,
                InitialTheta = THETA,
                ThetaFactorPerEpoch = epoch => 0.9999,
                ShuffleEveryEpoch = true
            };
            var learning = new Learning<Network, ILearningSample>(network, strategy, settings);
            await learning.Learn(samples);

            await network.Input(new double[] { 1, 0 });
            var afterLearning = (await network.Output()).First();

            Assert.True(beforeLearning < afterLearning);

            await network.Input(new double[] { 1, 0 });
            var output = (await network.Output()).First();
            Assert.True(Math.Abs(1 - output) < DELTA);

            await network.Input(new double[] { 1, 1 });
            output = (await network.Output()).First();
            Assert.True(Math.Abs(0 - output) < DELTA);

            await network.Input(new double[] { 0, 0 });
            output = (await network.Output()).First();
            Assert.True(Math.Abs(0 - output) < DELTA);

            await network.Input(new double[] { 0, 1 });
            output = (await network.Output()).First();
            Assert.True(Math.Abs(1 - output) < DELTA);
        }

        [Fact]
        public async Task TestCancel()
        {
            IInputLayer inputLayer = new InputLayer(() => new InputNode(), 2, new Bias());
            var innerLayer = new Layer(() => new Neuron(new Logistic(0.888)), 3, new Bias());
            var outputLayer = new Layer(new Neuron(new Logistic(0.777)));

            var network = new Network
            {
                InputLayer = inputLayer,
                OutputLayer = outputLayer
            };
            network.AddInnerLayer(innerLayer);

            var generator = new EachToEachSynapseGenerator(new Random());
            generator.Generate(network, inputLayer, innerLayer);
            generator.Generate(network, innerLayer, outputLayer);

            
            var samples = new List<ILearningSample>
            {
                new LearningSample(new double[] { 0, 1 }, new double[] { 1 }),
                new LearningSample(new double[] { 1, 0 }, new double[] { 1 }),
                new LearningSample(new double[] { 0, 0 }, new double[] { 0 }),
                new LearningSample(new double[] { 1, 1 }, new double[] { 0 })
            };

            var strategy = new BackpropagationStrategy();
            var settings = new LearningSettings { EpochRepeats = 20000 };
            var learning = new Learning<Network, ILearningSample>(network, strategy, settings);

            var cts = new CancellationTokenSource();
            var task = Task.Run(async () =>
            {
                await Task.Delay(1000);
                cts.Cancel();
            });

            await Assert.ThrowsAsync<OperationCanceledException>(async () => await learning.Learn(samples, cts.Token));
        }

        [Fact]
        public async Task TestTeachLite()
        {
            var inputLayer = new InputLayer(new InputNode(), new Bias());
            var innerLayer = new Layer(new Neuron(new Rectifier()));
            var outputLayer = new Layer(new Neuron(new Rectifier()));

            var network = new Network
            {
                InputLayer = inputLayer,
                OutputLayer = outputLayer
            };
            network.AddInnerLayer(innerLayer);

            var generator = new EachToEachSynapseGenerator(new Random());
            generator.Generate(network, inputLayer, innerLayer);
            generator.Generate(network, innerLayer, outputLayer);

            var samples = new List<ILearningSample>
            {
                new LearningSample(new double[] { 0 }, new double[] { 1 }),
            };

            var strategy = new BackpropagationStrategy();
            var settings = new LearningSettings
            {
                EpochRepeats = 10000,
                InitialTheta = THETA,
                ThetaFactorPerEpoch = epoch => 0.9995
            };
            var learning = new Learning<Network, ILearningSample>(network, strategy, settings);

            await learning.Learn(samples);

            await network.Input(new double[] { 1 });
            var output = (await network.Output()).First();
            Assert.True(Math.Abs(output) < DELTA);
        }

        private class EachToEachSynapseGenerator
        {

            private readonly Random _random;

            public EachToEachSynapseGenerator() => _random = new Random();

            public EachToEachSynapseGenerator(Random random) => _random = random;

            public void Generate(ISimpleNetwork network, IReadOnlyLayer<INode> masterLayer, IReadOnlyLayer<INotInputNode> slaveLayer)
            {
                foreach (var mNode in masterLayer.Nodes)
                {
                    foreach (var sNode in slaveLayer.Nodes.OfType<ISlaveNode>())
                    {
                        var weight = _getRandomWeight();
                        var synapse = new Synapse(mNode, sNode, weight);

                        network.AddSynapse(synapse);
                    }
                }
            }

            private double _getRandomWeight() => (_random.NextDouble() - 0.5) * 2;

        }
    }
}
