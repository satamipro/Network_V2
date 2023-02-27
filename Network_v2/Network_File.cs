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
        //ネットワークの構造と重みを保存する
        public void Save(string filename)
        {
            using(BinaryWriter save = new BinaryWriter(File.OpenWrite(filename + ".Network")))
            {
                //配列addressの書き込み
                save.Write(address.Count);
                for (int i = 0; i < address.Count; i++)
                {
                    save.Write(address[i]);
                }

                //配列nodeの書き込み
                save.Write(node.Count);
                for (int i = 0; i < node.Count; i++)
                {
                    save.Write(node[i].Length);
                }

                //配列denseの書き込み
                save.Write(dense.Count);
                for (int i = 0; i < dense.Count; i++)
                {
                    save.Write(dense[i].GetLength(0));
                    save.Write(dense[i].GetLength(1));
                    for (int j = 0; j < dense[i].GetLength(0); j++)
                    {
                        for (int k = 0; k < dense[i].GetLength(1); k++)
                        {
                            save.Write(dense[i][j, k]);
                        }
                    }
                }

                //バイアスの書き込み
                save.Write(bias.Count);
                for (int i = 0; i < bias.Count; i++)
                {
                    save.Write(bias[i].Length);
                    for (int j = 0; j < bias[i].Length; j++)
                    {
                        save.Write(bias[i][j]);
                    }
                }
            }
        }

        //ネットワークの構造と重みをロードする。
        public static Network Load(string  filename)
        {
            //変数定義
            int cnt1, cnt2, cnt3;
            Network ans = new Network();

            //ファイル読み込み
            using(BinaryReader read = new BinaryReader(File.OpenRead(filename)))
            {
                //address配列の読み込み
                cnt1 = read.ReadInt32();
                for (int i = 0; i < cnt1; i++)
                {
                    ans.address.Add(read.ReadInt32());
                }

                //node配列の読み込み
                cnt1 = read.ReadInt32();
                ans.num_layer = cnt1;
                for(int i = 0; i < cnt1; i++)
                {
                    cnt2 = read.ReadInt32();
                    ans.length.Add(cnt2);
                    ans.node.Add(new Double[cnt2]);
                    ans.delta.Add(new Double[cnt2]);
                    ans.deltasum.Add(new Double[cnt2]);
                }

                //dense配列の読み込み
                cnt1 = read.ReadInt32();
                for (int i = 0; i < cnt1; i++)
                {
                    cnt2 = read.ReadInt32();
                    cnt3 = read.ReadInt32();
                    ans.dense.Add(new Double[cnt2, cnt3]);
                    for (int j = 0; j < cnt2; j++)
                    {
                        for (int k = 0; k < cnt3; k++)
                        {
                            ans.dense[i][j, k] = read.ReadDouble();
                        }
                    }
                }

                //バイアスの読み込み
                cnt1 = read.ReadInt32();
                for (int i = 0; i < cnt1; i++)
                {
                    cnt2 = read.ReadInt32();
                    ans.bias.Add(new Double[cnt2]);
                    for (int j = 0; j < cnt2; j++)
                    {
                        ans.bias[i][j] = read.ReadDouble();
                    }
                }
            }
            return ans;
        }
    }
}