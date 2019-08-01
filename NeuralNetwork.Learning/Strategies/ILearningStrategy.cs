using NeuralNetwork.Learning.Samples;
using NeuralNetwork.Structure.Contract.Networks;
using System.Threading.Tasks;

namespace NeuralNetwork.Learning.Strategies
{
    public interface ILearningStrategy<in TNetwork, in TSample>
        where TNetwork : ISimpleNetwork
        where TSample : ISample
    {

        Task LearnSample(TNetwork network, TSample sample, double theta);

    }
}