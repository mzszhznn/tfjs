// 导入 TensorFlow.js 和 Plotly.js 库
const tf = require('@tensorflow/tfjs-node');
const plot = require('node-remote-plot');
// 准备数据集，假设有 100 个样本点
const area = tf.linspace(0, 2500, 100); // 房屋面积，从 0 到 2500 平方英尺
const price = area.mul(300).add(tf.randomNormal([100], 0, 40000)); // 房屋价格，假设有一个基础价格和一些随机噪声

// 划分数据集为训练集和测试集，假设按照 8:2 的比例划分
const splitIndex = Math.floor(0.8 * area.size);
const [areaTrain, areaTest] = tf.split(area, [splitIndex, -1]);
const [priceTrain, priceTest] = tf.split(price, [splitIndex, -1]);

// 创建一个线性回归模型，并初始化参数
const model = tf.sequential(); // 创建一个顺序模型
model.add(tf.layers.dense({ units: 1, inputShape: [1] })); // 添加一个全连接层，输出单元为 1，输入形状为 [1]
// model.compile({ optimizer: tf.train.sgd(0.00001), loss: 'meanSquaredError' }); // 编译模型，指定优化器为 SGD，损失函数为 MSE
model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' }); // 编译模型，指定优化器为 Adam，损失函数为 Huber

// 训练模型，并记录损失和准确率
const trainModel = async () => {
  const history = []; // 存储每一轮训练的损失值

  for (let i = 0; i < 50; i++) { // 假设训练 50 轮
    const result = await model.fit(areaTrain, priceTrain, { epochs: 10 }); // 每轮训练使用全部训练集，并迭代 10 次 
    const lossValue = result.history.loss[0]; // 获取当前轮的损失值
    history.push(lossValue); // 将损失值添加到历史记录中

    console.log(`Epoch ${i}: lossValue=${lossValue}`); // 打印当前轮的损失值

    await plot({ // 使用 Plotly.js 库来可视化历史记录中的损失值变化曲线
      x: history,
      xLabel: 'Iteration',
      yLabel: 'Loss'
    });
  }
};



// 评估模型，并可视化结果
const evaluateModel = async () => {
  const preds = model.predict(areaTest); // 使用模型对测试集进行预测
  const predsArray = await preds.array(); // 将预测结果转换为数组
  const labelsArray = await priceTest.array(); // 将真实标签转换为数组

  let totalError = 0; // 计算总误差
  for (let i = 0; i < predsArray.length; i++) {
    totalError += Math.abs(predsArray[i] - labelsArray[i]); // 累加每个样本点的绝对误差
  }
  const meanError = totalError / predsArray.length; // 计算平均误差

  console.log(`Mean error: ${meanError}`); // 打印平均误差

  await plot({ // 使用 Plotly.js 库来可视化测试集中的真实价格和预测价格的散点图
    x: labelsArray.flat(),
    y: predsArray.flat(),
    xLabel: 'Actual Price',
    yLabel: 'Predicted Price',
    type: 'scatter'
  });
};
trainModel() // 调用训练函数
  .then(() => {
    evaluateModel(); // 调用评估函数
  })