using NeuralNetwork.Learning.Samples;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetwork.Learning
{
    public interface ILearning<in TSample>
        where TSample : ISample
    {
        Task Learn(IEnumerable<TSample> samples, CancellationToken ct = default(CancellationToken));
    }
}