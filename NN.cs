using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;

namespace NNUPDATE
{
	[Serializable]
	class NN
	{
		private double learningRate = 0.00015;

		private List<List<double>> biases;
		private List<List<double>> errors;
		private List<List<double>> values;
		private List<List<List<double>>> weights;

		private Random r = new Random();

		public NN(int[] layers)
		{
			initNeurons(layers);
			initWeights(layers);
		}

		private void initNeurons(int[] layers)
		{
			biases = new List<List<double>>();
			errors = new List<List<double>>();
			values = new List<List<double>>();

			for (int layerIdx = 0; layerIdx < layers.Length; layerIdx++)
			{
				biases.Add(new List<double>());
				errors.Add(new List<double>());
				values.Add(new List<double>());

				for (int neuronIdx = 0; neuronIdx < layers[layerIdx]; neuronIdx++)
				{
					biases[layerIdx].Add(0);
					errors[layerIdx].Add(0);
					values[layerIdx].Add(0);
				}
			}
		}

		private void initWeights(int[] layers)
		{
			weights = new List<List<List<double>>>();

			for (int layerIdx = 0; layerIdx < layers.Length - 1; layerIdx++)
			{
				weights.Add(new List<List<double>>());

				for (int neuronIdx = 0; neuronIdx < layers[layerIdx]; neuronIdx++)
				{
					weights[layerIdx].Add(new List<double>());

					for (int nextNeuronIdx = 0; nextNeuronIdx < layers[layerIdx + 1]; nextNeuronIdx++)
					{
						weights[layerIdx][neuronIdx].Add(r.NextDouble() - 0.5f);
					}
				}
			}
		}

		private void activateTanh()
		{
			for (int layerIdx = 1; layerIdx < values.Count(); layerIdx++)
			{
				for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
				{
					double sum = 0;
					for (int prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].Count(); prevNeuronIdx++)
					{
						sum += values[layerIdx - 1][prevNeuronIdx] * weights[layerIdx - 1][prevNeuronIdx][neuronIdx];
					}
					values[layerIdx][neuronIdx] = Math.Tanh(sum + biases[layerIdx][neuronIdx]);
				}
			}
		}

		public double[] getTanhOutput(double[] inputs)
		{
			System.Diagnostics.Contracts.Contract.Requires(inputs.Length == values[0].Count());

			for (int neuronIdx = 0; neuronIdx < inputs.Length; neuronIdx++)
			{
				values[0][neuronIdx] = inputs[neuronIdx];
			}

			activateTanh();

			return values[values.Count() - 1].ToArray();
		}

		public void backPropagateTanh(double[] correctOutput)
		{
			for (int neuronIdx = 0; neuronIdx < values[values.Count() - 1].Count(); neuronIdx++)
			{
				errors[errors.Count() - 1][neuronIdx] = (correctOutput[neuronIdx] - values[values.Count() - 1][neuronIdx]) * (1 - values[values.Count() - 1][neuronIdx] * values[values.Count() - 1][neuronIdx]);
			}

			for (int layerIdx = values.Count() - 2; layerIdx > 0; layerIdx--)
			{
				for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
				{
					double error = 0;
					for (int nextNeuronIdx = 0; nextNeuronIdx < values[layerIdx + 1].Count(); nextNeuronIdx++)
					{
						error += errors[layerIdx + 1][nextNeuronIdx] * weights[layerIdx][neuronIdx][nextNeuronIdx];
					}
					errors[layerIdx][neuronIdx] = error * (1 - values[layerIdx][neuronIdx] * values[layerIdx][neuronIdx]);
				}
			}

			for (int layerIdx = values.Count() - 1; layerIdx > 0; layerIdx--)
			{
				for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
				{
					biases[layerIdx][neuronIdx] += errors[layerIdx][neuronIdx] * learningRate;
					for (int prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].Count(); prevNeuronIdx++)
					{
						weights[layerIdx - 1][prevNeuronIdx][neuronIdx] += values[layerIdx - 1][prevNeuronIdx] * errors[layerIdx][neuronIdx] * learningRate;
					}
				}
			}
		}

		public void TrainTanh(double[] inputs, double[] correctOutput)
		{
			getTanhOutput(inputs);
			backPropagateTanh(correctOutput);
		}

		public void Optimize(int[] layers, double[][] trainingData, double[][] correctOutputs, double[][] testingData, double[][] correctTests)
		{
			learningRate = 1;
			initNeurons(layers);
			initWeights(layers);
			double error = 0;
			int count = 0;
			while (error != -1)
			{
				for (int i = 0; i < 1000; i++)
				{
					for (int j = 0; j < trainingData.Length; j++)
					{
						TrainTanh(trainingData[j], correctOutputs[j]);
					}
				}
				error = getError(trainingData, correctOutputs);
				for (int i = 0; i < 1000; i++)
				{
					for (int j = 0; j < trainingData.Length; j++)
					{
						TrainTanh(trainingData[j], correctOutputs[j]);
					}
				}
				if (error <= getError(trainingData, correctOutputs))
				{
					count++;
				}
				if (count > 5)
				{
					learningRate *= 0.1;
					count = 6;
				}
				if (count > 10)
				{
					break;
				}
				if (Accurate(testingData, correctTests))
				{
					Save(String.Join(",",layers));
					break;
				}
			}
			if (Accurate(trainingData, correctOutputs))
			{
				Save(String.Join(",",layers));
				return;
			}
			if (layers[1] >= layers[0] * 2)
			{
				int[] newList = new int[layers.Length + 1];
				for (int i = 0; i < layers.Length - 1; i++)
				{
					newList[i] = layers[i];
				}
				newList[layers.Length] = layers[layers.Length];
				newList[layers.Length - 1] = 2;
				Optimize(newList, trainingData, correctOutputs, testingData, correctTests);
			}
			else
			{
				if (layers[layers.Length - 1] < layers[layers.Length - 2])
				{
					layers[layers.Length - 1]++;
					Optimize(layers, trainingData, correctOutputs, testingData, correctTests);
				}
				else
				{
					int[] newList = new int[layers.Length + 1];
					for (int i = 0; i < layers.Length - 1; i++)
					{
						newList[i] = layers[i];
					}
					newList[layers.Length] = layers[layers.Length];
					newList[layers.Length - 1] = 2;
					Optimize(newList, trainingData, correctOutputs, testingData, correctTests);
				}
			}
		}

		private double getError(double[][] trainingData, double[][] testingData)
		{
			double error = 0;
			for (int i = 0; i < trainingData.Length; i++)
			{
				double[] output = getTanhOutput(trainingData[i]);
				double[] correctOutput = testingData[i];
				for (int j = 0; j < output.Length; j++)
				{
					error += Math.Abs(output[j] - correctOutput[j]);
				}
			}
			return error;
		}

		public bool Accurate(double[][] data, double[][] correctOutputs)
		{
			int correct = 0;

			for (int i = 0; i < data.Length; i++)
			{
				double[] output = getTanhOutput(data[i]);
				double[] correctOutput = correctOutputs[i];
				bool isCorrect = true;

				for (int j = 0; j < output.Length; j++)
				{
					if (Math.Abs(output[j] - correctOutput[j]) > 0.5)
					{
						isCorrect = false;
						break;
					}
				}

				if (isCorrect)
				{
					correct++;
				}
			}

			return (correct / data.Length) > 0.9;
		}

		public void Save(string name)
		{
			using (FileStream fileStream = new FileStream(name, FileMode.Create))
			{
				System.Runtime.Serialization.Formatters.Binary.BinaryFormatter binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				binaryFormatter.Serialize(fileStream, this);
			}
		}

		public void Load(string name)
		{
			using (FileStream fileStream = new FileStream(name, FileMode.Open))
			{
				System.Runtime.Serialization.Formatters.Binary.BinaryFormatter binaryFormatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
				NN brain = (NN)binaryFormatter.Deserialize(fileStream);
				this.values = brain.values;
				this.weights = brain.weights;
				this.biases = brain.biases;
				this.errors = brain.errors;
			}
		}
	}
}
