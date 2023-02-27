using System.Threading.Tasks;

namespace Calculation
{
    partial class Calc
    {
        //全結合層(非同期)
        public static Double[] DenseAsync(Double[] input, Double[,] weight)
        {
            //配列長の整合性確認
            if (input.Length != weight.GetLength(0))
            {
                Console.WriteLine("Error:Calc.Dense  not match length of input.");
            }

            //変数定義
            Double[] ans = new Double[weight.GetLength(1)];
            
            Parallel.For(0, weight.GetLength(1), i => 
            {
                ans[i] = 0;
                for (int j = 0; j < weight.GetLength(0); j++)
                {
                    ans[i] += input[j] * weight[j, i];
                }
            });
            return ans;
        }

        //全結合層の偏微分
        public static Double[] DenseBackAsync(Double[] input, Double[,] weight)
        {
            //配列の長さチェック
            if (input.Length != weight.GetLength(1))
            {
                Console.WriteLine("Error:Calc.DenseBack  not match length of input.");
            }
            //変数定義
            Double[] ans = new Double[weight.GetLength(0)];
            
            //計算処理
            Parallel.For (0, weight.GetLength(0), i =>
            {
                ans[i] = 0;
                for (int j = 0; j < weight.GetLength(1); j++)
                {
                    ans[i] += input[j] * weight[i, j];
                }
            });
            /*
            for (int i = 0; i < weight.GetLength(0); i++)
            {
                ans[i] = 0;
                for (int j = 0; j < weight.GetLength(1); j++)
                {
                    ans[i] += input[j] * weight[i, j];
                }
            }
            */
            return ans;
        }

        //バイアスあり全結合層
        public static Double[] DenseBAsync(Double[] input, Double[,] weight, Double[] bias)
        {
            //配列長の整合性確認
            if (input.Length != weight.GetLength(0))
            {
                Console.WriteLine("Error:Calc.DenseB  not match length of input.");
                Console.WriteLine("input.Length:{0}, weight.GetLength(0):{1}, input[{0}] = {2}",input.Length, weight.GetLength(0), input[34]);
            }
            if (bias.Length != weight.GetLength(1))
            {
                Console.WriteLine("Error:Calc.DenseB  not match length of input.1");
            }

            //変数定義
            Double[] ans = new Double[weight.GetLength(1)];
            
            //計算処理
            Parallel.For (0, weight.GetLength(1), i =>
            {
                ans[i] = 0; 
                for (int j = 0; j < weight.GetLength(0); j++)
                {
                    ans[i] += input[j] * weight[j, i];
                }
                ans[i] += bias[i];
            });
            /*
            for (int i = 0; i < weight.GetLength(1); i++)
            {

                ans[i] = 0;
                for (int j = 0; j < weight.GetLength(0); j++)
                {
                    ans[i] += input[j] * weight[j, i];
                }
                ans[i] += bias[i];
            }
            */
            return ans;
        }
    
        //ReLU関数
        public static Double[] ReLUAsync(Double[] input)
        {
            //変数定義
            Double[] ans = new Double[input.Length];

            //計算処理
            Parallel.For (0, input.Length, i =>
            {
                if (input[i] < 0)
                {
                    ans[i] = 0;
                } else 
                {
                    ans[i] = input[i];
                }
            });
            return ans;
        }
    
        //ReLUの偏微分
        public static Double[] ReLUBackAsync(Double[] input, Double[] node)
        {
            //配列の長さ確認
            if (input.Length != node.Length)
            {
                Console.WriteLine("Error:Calc.ReLUBack  not match length of input.");
            }

            //変数定義
            Double[] ans = new Double[input.Length];

            //計算処理

            Parallel.For (0, ans.Length, i =>
            {
                if (node[i] < 0)
                {
                    ans[i] = 0;
                } else
                {
                    ans[i] = input[i];
                }
            });
            return ans;
        }
    
        //Softmax関数
        public static Double[] SoftmaxAsync(Double[] input)
        {
            //変数定義
            double sum = 0;
            Double[] ans = new Double[input.Length];

            //計算処理
            Parallel.For (0, input.Length, i =>
            {
                ans[i] = Math.Exp(input[i]);
            });
            for (int i = 0; i < input.Length; i++)
            {
                sum += ans[i];
            }
            Parallel.For (0, input.Length, i =>
            {
                ans[i] /= sum;
            });
            return ans;
        }
    
        //Softmax関数の偏微分
        public static Double[] SoftmaxBackAsync(Double[] input, Double[] node)
        {
            //配列の長さ確認
            if (input.Length != node.Length)
            {
                Console.WriteLine("Error:Calc.SoftmaxBack  not match length of input.");
            }
            //変数定義
            Double[] ans = new Double[input.Length];

            //計算処理
            Parallel.For (0, node.Length, i =>
            {
                ans[i] = 0;
                for (int j = 0; j < node.Length; j++)
                {
                    if (i == j)
                    {
                        //Console.WriteLine("{0}", tmp);
                        ans[i] += input[j] * node[i] * (1 - node[j]);
                    } else 
                    {
                        //Console.WriteLine("{0}", tmp);
                        ans[i] -= input[j] * node[i] * node[j];
                    }
                }
            });
            return ans;
        }
    
        //NormalizationMM関数
        public static Double[] NormalizationMMAsync(Double[] input)
        {
            //変数定義
            double min = input[0], max = input[0], range;
            Double[] ans = new Double[input.Length];

            //計算処理
            for (int i = 0; i < input.Length; i++)
            {
                if (min > input[i])
                {
                    min = input[i];
                }
                if (max < input[i])
                {
                    max = input[i];
                }
            }
            range = max - min;
            Parallel.For (0, input.Length, i =>
            {
                ans[i] = (input[i] - min) / range;
            });
            return ans;
        }
    
        //NormalizationMM関数の偏微分
        public static Double[] NormalizationMMBackAsync(Double[] input, Double[] node)
        {
            //変数定義
            int maxindex = 0, minindex = 0;
            double max = node[0], min = node[0], range;
            Double[] ans = new Double[input.Length];

            //計算処理
            for (int i = 0; i < input.Length; i++)
            {
                if (max < node[i])
                {
                    max = node[i];
                    maxindex = i;
                } else if (min > node[i])
                {
                    min = node[i];
                    minindex = i;
                }
            }
            range = max - min;
            Parallel.For (0, input.Length, i =>
            {
                if (i == maxindex)
                {
                    ans[i] = 0;
                } else if (i == minindex)
                {
                    ans[i] = 0;
                } else 
                {
                    ans[i] = input[i] / range;
                }
            });
            return ans;
        }
    
        //NormalizationZ関数
        public static Double[] NormalizationZAsync(Double[] input)
        {
            //変数定義
            double sum = 0, ave = 0, dev = 0;
            Double[] ans = new Double[input.Length];
            //計算処理
            for (int i = 0; i < input.Length; i++)
            {
                sum += input[i];
            }
            ave = sum / input.Length;
            sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                sum += Math.Pow(input[i] - ave, 2);
            }
            dev = Math.Sqrt(sum / input.Length);

            Parallel.For (0, input.Length, i =>
            {
                ans[i] = (input[i] - ave) / dev;
            });
            return ans;
        }
    
        //NormalizationZ関数の偏微分
        public static Double[] NormalizationZBackAsync(Double[] input, Double[] node)
        {
            //変数定義
            double sum = 0, ave = 0, dev = 0;
            Double[] ans = new Double[input.Length];
            //計算処理
            for (int i = 0; i < input.Length; i++)
            {
                sum += node[i];
            }
            ave = sum / input.Length;
            sum = 0;
            for (int i = 0; i < input.Length; i++)
            {
                sum += Math.Pow(node[i] - ave, 2);
            }
            dev = Math.Sqrt(sum / input.Length);

            Parallel.For (0, input.Length, i =>
            {
                ans[i] = input[i] * (1 - (1 / input.Length)) * (1 / dev * dev) * (dev - ((1 / input.Length) * Math.Pow(node[i] - ave, 2) / dev));
            });
            return ans;
        }
    }
}