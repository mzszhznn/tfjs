// const tf = require('@tensorflow/tfjs-node');
// const model = tf.sequential();
// model.add(tf.layers.dense({ units: 1, inputShape: [1] }));
// model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
// const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
// const ys = tf.tensor2d([1, 3, 5,2], [4, 1]);
// const logdir = './logs';
// const tensorBoardCallback = tf.node.tensorBoard(logdir);
// model.fit(xs, ys, { epochs: 10, callbacks: tensorBoardCallback })
//   .then(() => {
//     model.predict(xs).print()
//   })

const tf = require('@tensorflow/tfjs-node');

const csvUrl =
  'https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/boston-housing-train.csv';

async function run() {
  const logdir = './logs';
  const tensorBoardCallback = tf.node.tensorBoard(logdir);
  // We want to predict the column "medv", which represents a median value of
  // a home (in $1000s), so we mark it as a label.
  const csvDataset = tf.data.csv(
    csvUrl, {
    columnConfigs: {
      medv: {
        isLabel: true
      }
    }
  });

  // Number of features is the number of column names minus one for the label
  // column.
  const numOfFeatures = (await csvDataset.columnNames()).length - 1;
  let test;
  // Prepare the Dataset for training.
  const flattenedDataset =
    csvDataset
      .map(({ xs, ys }) => {
        // Convert xs(features) and ys(labels) from object form (keyed by
        // column name) to array form.
        test = Object.values(xs);
        return { xs: Object.values(xs), ys: Object.values(ys) };
      })
      .batch(10);

  // Define the model.
  const model = tf.sequential();
  model.add(tf.layers.dense({
    inputShape: [numOfFeatures],
    units: 1
  }));
  model.compile({
    optimizer: tf.train.sgd(0.000001),
    loss: 'meanSquaredError'
  });

  // Fit the model using the prepared Dataset
  return model.fitDataset(flattenedDataset, {
    epochs: 10,
    // callbacks: {
    //   onEpochEnd: async (epoch, logs) => {
    //     console.log(epoch + ':' + logs.loss);
    //   },
    //   onTrainBegin: async (logs) => {
    //     console.log("Train Begin!");
    //   },
    //   onTrainEnd: async (logs) => {
    //     console.log('Train End!');
    //   }
    // }
    callbacks: tensorBoardCallback
  })
    .then(() => {
      tf.tensor(test).expandDims(0).print();
      const preds = model.predict(tf.tensor(test).expandDims(0)); // 使用模型对测试集进行预测
      preds.print();
    });
}

run();
