using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Diagnostics;
using Calculation;

namespace NeuralNetwork
{
    partial class Network
    {
        //計算の種類を指定する。 address[i]にnode[i]からnode[i + 1]の計算手法が書かれている。
        //0:dense, 1:dense(バイアスあり), 128:ReLU, 256:Softmax, 257:NormalizationZ
        List<int> address = new List<int>();

        //ノードを追加する。
        public List<Double[]> node = new List<Double[]>();

        //node[i], delta[i]の要素数をlength[i]に格納する。
        List<int> length = new List<int>();

        //全結合層で使用する重みを追加する。node[i]からnode[i + 1]に使用する重みをdense[i]に格納する。
        List<Double[,]> dense = new List<Double[,]>();

        //バイアスの値を保存する。bias[i]に、node[i]に使用するバイアスが保存されている。
        List<Double[]> bias = new List<Double[]>();

        //node[i][j]の誤差関数の偏微分の値をdelta[i][j]に格納する。
        List<Double[]> delta = new List<Double[]>();

        //deltaの和を保存する。
        List<Double[]> deltasum = new List<Double[]>();

        //Networkの深さがここに格納される。
        int num_layer = 0;

        //コンストラクタ
        public Network()
        {

        }

        //入力層を追加する。
        public void AddInput(int num_of_input)
        {
            node.Add(new Double[num_of_input]);
            bias.Add(new Double[0]);
            length.Add(num_of_input);
            delta.Add(new Double[num_of_input]);
            deltasum.Add(new Double[num_of_input]);
            num_layer = 1;
        }
        //Networkに関する情報を表示する。
        public void ShowInfo()
        {
            Console.WriteLine("Network Infomation");
            Console.WriteLine(" layer list");
            for (int i = 0; i < address.Count; i++)
            {
                Console.WriteLine(" ・layer{0}  num_node:{1}", i, node[i].Length);
                switch (address[i])
                {
                    case 0:
                    case 1:
                        Console.Write(" ・Calctype{0}→{1}:Dense", i, i + 1);
                        Console.WriteLine("  size: [{0}, {1}]", dense[i].GetLength(0), dense[i].GetLength(1));
                        break;

                    case 128:
                        Console.WriteLine(" ・Calctype{0}→{1}:ReLU", i, i + 1);
                        break;

                    case 129:
                        Console.WriteLine(" ・Calctype{0}→{1}:Normalization", i, i + 1);
                        break;

                    case 256:
                        Console.WriteLine(" ・Calctype{0}→{1}:Softmax", i, i + 1);
                        break;

                    case 257:
                        Console.WriteLine(" ・Calctype{0}→{1}:NormalizationZ", i, i + 1);
                        break;

                }
            }
        }

        //直前に入力された値を表示する。
        public void ShowInput()
        {
            Console.Write("Input----");
            for (int i = 0; i < node[0].Length; i++)
            {
                Console.Write("{0}:{1:0.00000}, ", i + 1, node[0][i]);
            }
            Console.WriteLine();
        }

        //任意の層の値を表示する。
        public void ShowLayer(int i)
        {
            for (int j = 0; j < length[i]; j++)
            {
                Console.Write("{0}:{1:0.00000}, ", j, node[i][j]);
            }
            Console.WriteLine("");
        }

        //任意の層のバイアスを表示する。
        public void ShowBias(int i)
        {
            Console.WriteLine("Bias.Layer{0} ", i);
            for (int j = 0; j < bias[i].Length; j++)
            {
                Console.WriteLine("{0}:{1}, ", j, bias[i][j]);
            }
        }

        //演算結果を表示する。
        public void ShowOutput()
        {
            Console.Write("Output---");
            for (int i = 0; i < node[node.Count - 1].Length; i++)
            {
                Console.Write("{0}:{1:0.00}, ", i + 1, node[node.Count - 1][i]);
            }
            Console.WriteLine();
        }

        //直前に計算された偏微分値を表示する。
        public void ShowDelta(int k)
        {
            for (int i = 0; i < delta[k].Length; i++)
            {
                Console.Write("{0}:{1}, ", i, delta[k][i]);
            }
            Console.WriteLine();
        }

        //恒等写像な層を追加する。    
        private void AddNormalLayer(int i)
        {
            address.Add(-2);
            node.Add(new Double[i]);
            bias.Add(new Double[0]);
            length.Add(i);
            dense.Add(new Double[0, 0]);
            delta.Add(new Double[i]);
            deltasum.Add(new Double[i]);
            num_layer++;
        }

        //全結合層を追加する。
        public void AddDense(int i, bool addbias)
        {
            AddNormalLayer(i);
            if (addbias == true)
            {
                address[num_layer - 2] = 1;
                bias[num_layer - 1] = new Double[i];
            } 
            else 
            {
                address[num_layer - 2] = 0;
            }
            
            dense[num_layer - 2] = new Double[length[num_layer - 2], i];
            
        }

        //ReLU関数を追加する。  
        public void AddReLU()
        {
            AddNormalLayer(length[num_layer - 1]);
            address[num_layer - 2] = 128;
        }

        //Softmax関数を追加する。
        public void AddSoftmax()
        {
            AddNormalLayer(length[num_layer - 1]);
            address[num_layer - 2] = 256;
        }

        //NormalixationMM関数を追加する
        public void AddNormalizationMM()
        {
            AddNormalLayer(length[num_layer - 1]);
            address[num_layer - 2] = 258;
        }

        //NormalizationZ関数を追加する
        public void AddNormalizationZ()
        {
            AddNormalLayer(length[num_layer - 1]);
            address[num_layer - 2] = 257;
        }

        //Networkのすべての重みの値をlower以上upper以下の値で初期化。
        public void InitWeight(double lower, double upper)
        {
            //変数定義
            double range = upper - lower;
            Random rand = new Random();
            //計算処理
            for (int i = 0; i < address.Count; i++)
            {
                switch (address[i])
                {
                    case 0:
                        for (int j = 0; j < dense[i].GetLength(0); j++)
                        {
                            for (int k = 0; k < dense[i].GetLength(1); k++)
                            {
                                dense[i][j, k] = rand.NextDouble() * range + lower;
                            }
                        }
                        break;
                    
                    case 1:
                        for (int j = 0; j < dense[i].GetLength(0); j++)
                        {
                            for (int k = 0; k < dense[i].GetLength(1); k++)
                            {
                                dense[i][j, k] = rand.NextDouble() * range + lower;
                            }
                        }
                        for (int j = 0; j < dense[i].GetLength(1); j++)
                        {
                            bias[i + 1][j] = rand.NextDouble() * range + lower;
                        }
                        break;

                }
            }
        }

        //出力値を計算する。
        public Double[] Compute(Double[] input)
        {
            //入力データの長さ確認
            if (input.Length != node[0].Length)
            {
                Console.WriteLine("Error:Network.Compute  not match Length of input");
            }

            //計算処理
            node[0] = input;
            for (int i = 0; i < address.Count; i++)
            {
                switch (address[i])
                {
                    case 0:
                        node[i + 1] = Calc.DenseAsync(node[i], dense[i]);
                        break;

                    case 1:
                        node[i + 1] = Calc.DenseBAsync(node[i], dense[i], bias[i + 1]);
                        break;

                    case 128:
                        node[i + 1] = Calc.ReLUAsync(node[i]);
                        break;

                    case 256:
                        node[i + 1] = Calc.SoftmaxAsync(node[i]);
                        break;

                    case 257:
                        node[i + 1] = Calc.NormalizationZAsync(node[i]);
                        break;
                    
                    case 258:
                        node[i + 1] = Calc.NormalizationMMAsync(node[i]);
                        break;
                }
            }
            return node[num_layer - 1];
        }

        //偏微分値を計算する。
        public double BackPropagation(Double[] teachdata)
        {
            //教師データの長さ確認
            if (teachdata.Length != length[num_layer - 1])
            {
                Console.WriteLine("Error:Network.BackPropagation  not match length of input");
            }

            //変数定義
            double ans = 0;

            //計算処理
            //出力値と教師データとの差分を計算する。
            for (int i = 0; i < length[num_layer - 1]; i++)
            {
                delta[num_layer - 1][i] = node[num_layer - 1][i] - teachdata[i];
            }
            //中間層の偏微分値を計算する。
            for (int i = num_layer - 2; i >= 0; i--)
            {
                switch (address[i])
                {
                    case 0:
                    case 1:
                        delta[i] = Calc.DenseBackAsync(delta[i + 1], dense[i]);
                        break;

                    case 128:
                        delta[i] = Calc.ReLUBackAsync(delta[i + 1], node[i + 1]);
                        break;

                    case 256:
                        delta[i] = Calc.SoftmaxBackAsync(delta[i + 1], node[i + 1]);
                        break;

                    case 257:
                        delta[i] = Calc.NormalizationZBackAsync(delta[i + 1], node[i]);
                        break;

                    case 258:
                        delta[i] = Calc.NormalizationMMBackAsync(delta[i + 1], node[i]);
                        break;
                        

                }
            }

            //損失関数の値を計算
            for (int i = 0; i < length[num_layer - 1]; i++)
            {
                ans += Math.Pow(delta[num_layer - 1][i], 2);
            }
            return ans;
            
        }

        //偏微分値を累積する。
        public void BackPropagationAdd()
        {
            for (int i = 0; i < num_layer; i++)
            {
                for (int j = 0; j < delta[i].Length; j++)
                {
                    deltasum[i][j] += delta[i][j];
                }
            }
        }

        //偏微分値の累積をリセットする。
        public void ResetDelta()
        {
            for (int i = 0; i < num_layer; i++)
            {
                for (int j = 0; j < delta[i].Length; j++)
                {
                    deltasum[i][j] = 0;
                }
            }
        }

        //確率urateで、学習率lrateで重みとバイアスの値を更新する。
        public void UpdateWeight(double urate, double lrate)
        {
            Random rand = new Random();
            for (int i = 0; i < address.Count; i++)
            {
                switch (address[i])
                {
                    case 0:
                        for (int j = 0; j < dense[i].GetLength(0); j++)
                        {
                            for (int k = 0; k < dense[i].GetLength(1); k++)
                            {
                                if (urate > rand.NextDouble())
                                {
                                    dense[i][j, k] -= delta[i + 1][k] * node[i][j] * lrate;
                                }
                            }
                        }
                        break;

                    case 1:
                        for (int j = 0; j < dense[i].GetLength(0); j++)
                        {
                            for (int k = 0; k < dense[i].GetLength(1); k++)
                            {
                                if (urate > rand.NextDouble())
                                {
                                    dense[i][j, k] -= delta[i + 1][k] * node[i][j] * lrate;
                                }
                            }
                        }
                        for (int j = 0; j < dense[i].GetLength(1); j++)
                        {
                            if (urate > rand.NextDouble())
                            {
                                bias[i + 1][j] -= delta[i + 1][j] * lrate;
                            }
                        }
                        break;
                }
            }
        }

        public void ShowCompute()
        {
            Console.Write("input:          ");
            for (int i = 0; i < length[0]; i++)
            {
                Console.Write("{0:0.00}  ", node[0][i]);
            }
            Console.WriteLine("");
            for (int i = 0; i < length[1]; i++)
            {
                Console.Write("output:{0:0.0000}   ", node[1][i]);
                for (int j = 0; j < length[0]; j++)
                {
                    Console.Write("{0:0.00}  ", dense[0][i, j]);
                }
                Console.WriteLine();
            }
        }
    
        //出力が最大の箇所のラベルを取得する。
        public int GetMaxLabels()
        {
            //変数定義
            int ans = 0;
            double max = 0;
            
            //計算処理
            for (int i = 0 ; i < node[num_layer - 1].Length; i++)
            {
                if (max < node[num_layer - 1][i])
                {
                    max = node[num_layer - 1][i];
                    ans = i;
                }
            }
            return ans;
        }
    }
}