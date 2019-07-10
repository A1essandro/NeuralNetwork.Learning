using System.Collections.Generic;

namespace NeuralNetwork.Learning.Samples
{
    public interface ILearningSample : ISample
    {

        IEnumerable<double> Input { get; }

        IEnumerable<double> Output { get; }

    }
}