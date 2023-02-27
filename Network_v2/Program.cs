// See https://aka.ms/new-console-template for more information
using System;
using NeuralNetwork;
using MNIST;
using System.Diagnostics;

namespace Program
{
    class Program
    {
        static void Main()
        {
            //変数定義
            int maru;
            Stopwatch sw = new Stopwatch();
            TimeSpan ts;

            MNIST_Data mnist = new MNIST_Data();
            Network testnet = new Network();
            testnet.AddInput(mnist.Trd_GetImageSize());
            testnet.AddNormalizationZ();
            testnet.AddDense(1024, true);
            testnet.AddReLU();
            testnet.AddNormalizationMM();
            testnet.AddDense(10, true);
            testnet.AddSoftmax();
            testnet.InitWeight(0, 1);
            
            sw.Start();
            for (int i = 0; i <1; i++)
            {
                Console.WriteLine("{0}週目トレーニング開始", i + 1);
                for (int j = 0; j < 60000; j++)
                {
                    testnet.Compute(mnist.Trd_GetImage(j));
                    testnet.UpdateWeight(0.05, testnet.BackPropagation(mnist.Trl_GetLabel(j)));
                    if (j % 1000 == 0)
                    {
                        Console.Write("#");
                    }
                }
                Console.WriteLine();
                Console.WriteLine("{0}週目トレーニング終了", i + 1);
                maru = 0;
                
                for (int j = 0; j < 10000; j++)
                {
                    testnet.Compute(mnist.Ted_GetImage(j));
                    if (testnet.GetMaxLabels() == mnist.Tel_GetAns(j))
                    {
                        maru++;
                    }
                }
                Console.WriteLine("{0}週目テストデータ正解率：{1:N3}%", i + 1, (double)100 * ((double)maru / (double)10000));
                ts = sw.Elapsed;
                Console.WriteLine($"経過時間{ts}");
            }
        }      
    }
}
