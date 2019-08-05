using NeuralNetwork.Learning.Samples;
using NeuralNetwork.Learning.Strategies;
using NeuralNetwork.Structure.Contract.Networks;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetwork.Learning
{
    public class Learning<TNetwork, TSample> : ILearning<TSample>
        where TNetwork : ISimpleNetwork
        where TSample : ISample
    {

        public LearningSettings Settings { get; }

        private readonly TNetwork _network;
        private readonly ILearningStrategy<TNetwork, TSample> _strategy;
        private readonly LearningSettings _settings;

        public Learning(TNetwork network, ILearningStrategy<TNetwork, TSample> strategy, LearningSettings settings)
        {
            _network = network;
            _strategy = strategy;
            _settings = settings;
        }

        public async Task Learn(IEnumerable<TSample> samples, CancellationToken ct = default(CancellationToken))
        {
            var random = new Random();
            var theta = _settings.InitialTheta;
            for (var epoch = 0; epoch < _settings.EpochRepeats; epoch++)
            {
                if (_settings.ShuffleEveryEpoch)
                {
                    samples = samples.OrderBy(a => random.Next()).ToArray();
                }

                await _learnEpoch(samples.ToArray(), theta, ct).ConfigureAwait(false);

                theta *= _settings.ThetaFactorPerEpoch(epoch);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private async Task _learnEpoch(IEnumerable<TSample> samples, double theta, CancellationToken ct)
        {
            foreach (var sample in samples)
            {
                ct.ThrowIfCancellationRequested();
                await _strategy.LearnSample(_network, sample, theta).ConfigureAwait(false);
            }
        }

    }
}
