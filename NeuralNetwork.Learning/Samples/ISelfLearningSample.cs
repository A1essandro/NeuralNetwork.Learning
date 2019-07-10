using System.Collections.Generic;

namespace NeuralNetwork.Learning.Samples
{
    public interface ISelfLearningSample : ISample
    {

        IEnumerable<double> Input { get; }

    }
}