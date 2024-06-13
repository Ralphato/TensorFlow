

let savedString = "";
let predictions = "";
let retry = false;

document.getElementById('save').addEventListener('click', saveString);
document.getElementById('inpt').addEventListener('keypress', function(event){
  if(event.key == 'Enter'){
    saveString;
  }
});


function saveString(){
  
  const inputText = document.getElementById('inpt').value;
  savedString = inputText;
  predictions = "";
  measurementAvailable();
}




async function measurementAvailable(){
  let measurements = [
    'attachdetach_controller_forced_detaches', 'bootstrap_signer_rate_limiter_use', 'container_cpu_cfs_periods_total',
    'container_spec_cpu_period', 'container_spec_cpu_quota', 'container_spec_cpu_shares', 'machine_cpu_cores',
    'node_cpu_core_throttles_total', 'node_cpu_seconds_total', 'node_hwmon_chip_names', 'node_hwmon_temp_max_celsius',
    'process_cpu_seconds_total', 'scheduler_binding_duration_seconds_bucket', 'scheduler_binding_latency_microseconds_sum',
    'scheduler_e2e_scheduling_duration_seconds_bucket', 'scheduler_e2e_scheduling_latency_microseconds_sum',
    'scheduler_scheduling_algorithm_duration_seconds_bucket', 'scheduler_scheduling_latency_seconds_sum'
  ];
  if(savedString != " " && measurements.includes(savedString)){
    document.getElementById('results').innerText = 'Valid Input \n';
    console.log(`Valid Input. Measurement is ${savedString}`);
    await run(savedString);


    
    if(retry === true){
      console.log('Reavaluating');
      await reavaluate(savedString);
    }
     
    document.getElementById('results').innerText += predictions;
   
  }
  else{
    console.log('Invalid Input');
    document.getElementById('validity').innerText = 'Invalid Input \n';
  }
}



//document.addEventListener('DOMContentLoaded', run(savedString));
console.log('Hello TensorFlow');

/**
 * Get the car data reduced to just the variables we are interested
 * and cleaned of missing data.
 */
async function getData(measurementName) {
  const carsDataResponse = await fetch('https://gist.githubusercontent.com/Ralphato/088700e494c953a5ed01e5347601395c/raw/9868c04451612feb65ac882b9f6373f445533405/AllTraningData.JSON');
  const carsData = await carsDataResponse.json();
  const dataFiltered = carsData.filter(car => car.Measurement === measurementName);
  console.log(`Data used for training: ${JSON.stringify(dataFiltered, null, 2)}`);
  const cleaned = dataFiltered.map(car => ({
    sum: car.Sum,
    count: car.Count,
    label: car.Label,
  }))
  .filter(car => car.sum != null && car.count != null && car.label != null);

  const allSumsZero = cleaned.every(car => car.sum === 0);
  if(allSumsZero){
    console.log('All values = 0')
    document.getElementById('results').innerText = 'The measurement values are all 0';
    return null;
  }


  // Remove outliers using IQR method
  const sums = cleaned.map(car => car.sum);
  const { lowerBound, upperBound } = getIQRBounds(sums);
  const filteredData = cleaned.filter(car => car.sum >= lowerBound && car.sum <= upperBound);

  // Filter the data to include only labels 3 and 4
// const finalFiltered = filteredData.filter(car => car.label === 3 || car.label === 4);
// return finalFiltered;
  return filteredData;
  
}

/**
* Calculate IQR bounds for outlier detection.
*/
function getIQRBounds(values) {
  const sorted = values.slice().sort((a, b) => a - b);
  const q1 = sorted[Math.floor(sorted.length / 4)];
  const q3 = sorted[Math.floor(sorted.length * (3 / 4))];
  const iqr = q3 - q1;
  const lowerBound = q1 - 1.5 * iqr;
  const upperBound = q3 + 1.5 * iqr;
  return { lowerBound, upperBound };
}

async function run(measurementName) {

  
  // Load and plot the original input data that we are going to train on.
  const data = await getData(measurementName);
  if(data === null){
    return;
  }
  
  const values = data.map(d => ({
    x: d.sum, // use only 1 of the 2 inputs since it is a 2d graph
    y: d.label,
  }));

  tfvis.render.scatterplot(
    {name: 'Sum v Label'},
    {values},
    {
      xLabel: 'Sum',
      yLabel: 'Label',
      height: 300
    }
  );

  // Create the model
  const model = createModel();
  tfvis.show.modelSummary({name: 'Model Summary'}, model);

  // Convert the data to a form we can use for training.
  const tensorData = convertToTensor(data);
  const {inputs, labels} = tensorData;

  // Train the model
  await trainModel(model, inputs, labels);
  console.log('Done Training');
  // Make some predictions using the model and compare them to the
  // original data
  testModel(model, data, tensorData);
  await testRun(model, tensorData, measurementName);
}



function createModel() {
  // Create a sequential model
  const model = tf.sequential();

  // Add a single input layer
  model.add(tf.layers.dense({inputShape: [2], units: 50, activation: 'relu', useBias: true}));
  model.add(tf.layers.dense({units: 50, activation: 'sigmoid', useBias: true}));
  model.add(tf.layers.dense({units: 50, activation: 'sigmoid', useBias: true}));

  // Add an output layer
  model.add(tf.layers.dense({units: 1, useBias: true}));

  return model;
}

/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis.
 */
function convertToTensor(data) {
  // Wrapping these calculations in a tidy will dispose any
  // intermediate tensors.

  return tf.tidy(() => {
    // Step 1. Shuffle the data
    tf.util.shuffle(data);

    // Step 2. Convert data to Tensor
    const inputs = data.map(d => [d.sum, d.count]);
    const labels = data.map(d => d.label);

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 2]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    // Step 3. Normalize the data to the range 0 - 1 using min-max scaling
    const inputMax = inputTensor.max(0);
    const inputMin = inputTensor.min(0);
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      // Return the min/max bounds so we can use them later.
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

async function trainModel(model, inputs, labels) {
  // Prepare the model for training.
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ['mse'],
  });

  const batchSize = 32;
  const epochs = 200;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: 'Training Performance' },
      ['loss', 'mse'],
      { height: 200, callbacks: ['onEpochEnd'] }
    )
  });
}

function testModel(model, inputData, normalizationData) {
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  // Generate predictions for a range of numbers between 0 and 1 for both sum and count
  const [unNormSum, unNormCount, unNormPreds] = tf.tidy(() => {
    const sumNorm = tf.linspace(0, 1, 100);
    const countNorm = tf.linspace(0, 1, 100);
   
    // Combine the normalized sum and count into a 2D tensor
    const xsNorm = tf.stack([sumNorm, countNorm], 1);
    const predictions = model.predict(xsNorm);

    const unNormSum = sumNorm
      .mul(inputMax.gather(0).sub(inputMin.gather(0)))
      .add(inputMin.gather(0));

    const unNormCount = countNorm
      .mul(inputMax.gather(1).sub(inputMin.gather(1)))
      .add(inputMin.gather(1));

    const unNormPreds = predictions
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the data
    return [unNormSum.dataSync(), unNormCount.dataSync(), unNormPreds.dataSync()];
  });

  const predictedPointsSum = Array.from(unNormSum).map((val, i) => {
    return {x: val, y: unNormPreds[i]};
  });

  const originalPointsSum = inputData.map(d => ({
    x: d.sum, y: d.label,
  }));

  const predictedPointsCount = Array.from(unNormCount).map((val, i) => {
    return {x: val, y: unNormPreds[i]};
  });

  const originalPointsCount = inputData.map(d => ({
    x: d.count, y: d.label,
  }));

  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data (Sum)'},
    {values: [originalPointsSum, predictedPointsSum], series: ['original', 'predicted']},
    {
      xLabel: 'Sum',
      yLabel: 'Label',
      height: 300
    }
  );

  tfvis.render.scatterplot(
    {name: 'Model Predictions vs Original Data (Count)'},
    {values: [originalPointsCount, predictedPointsCount], series: ['original', 'predicted']},
    {
      xLabel: 'Count',
      yLabel: 'Label',
      height: 300
    }
  );
}

async function getTest(measurementName){
    const carsDataResponse = await fetch('https://gist.githubusercontent.com/Ralphato/4e90d9b146ba86d14c91364d888f50e0/raw/798d0802d5b762a8ad61cb038c59b5f07c31dd4d/AllTestingData');
    const carsData = await carsDataResponse.json();
    const carsDataMeas = carsData.filter(car => car.Measurement === measurementName);
    const cleaned = carsDataMeas.map(car => ({
      sum: car.Sum,
      count: car.Count,
      label: car.Label,
    }))
    .filter(car => car.sum != null && car.count != null && car.label != null);
    //let cleanedFiltered = cleaned.filter(car => car.label === 3 || car.label === 4);
    //return cleanedFiltered;

    return cleaned;
}

async function testRun(model, normalizationData, measurementName) {
  const test = await getTest(measurementName);

  // Normalize the test data using the same normalization parameters as the training data
  const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

  const testInputs = test.map(d => [d.sum, d.count]);
  const testLabels = test.map(d => d.label);

  const testInputTensor = tf.tensor2d(testInputs, [testInputs.length, 2]);
  const testLabelTensor = tf.tensor2d(testLabels, [testLabels.length, 1]);

  const normalizedTestInputs = testInputTensor.sub(inputMin).div(inputMax.sub(inputMin));
  const normalizedTestLabels = testLabelTensor.sub(labelMin).div(labelMax.sub(labelMin));

  // Make predictions on the normalized test data
  const testPredictions = model.predict(normalizedTestInputs);

  // Un-normalize the predictions
  const unNormTestPreds = testPredictions
    .mul(labelMax.sub(labelMin))
    .add(labelMin);

  // Un-normalize the test labels for comparison
  const unNormTestLabels = normalizedTestLabels
    .mul(labelMax.sub(labelMin))
    .add(labelMin);

  // Log the predicted and actual labels for the test data
  const unNormTestPredsArray = unNormTestPreds.dataSync();
  const unNormTestLabelsArray = unNormTestLabels.dataSync();

  console.log("Sum, Count, Predicted Label, Actual Label");
  //document.getElementById('results').innerText = 'Sum, Count, Predicted Label, Actual Label \n';
  for (let i = 0; i < unNormTestPredsArray.length; i++) {
    console.log(`Sum: ${test[i].sum}, Count: ${test[i].count}, Predicted Label: ${unNormTestPredsArray[i]}, Actual Label: ${unNormTestLabelsArray[i]}`);
    //document.getElementById('results').innerText += `Sum: ${test[i].sum}, Count: ${test[i].count}, Predicted Label: ${unNormTestPredsArray[i]}, Actual Label: ${unNormTestLabelsArray[i]} \n`;
  }

  let sortedResults = [];

// Populate the sortedResults array
for (let i = 0; i < unNormTestPredsArray.length; i++) {
  sortedResults.push({
    predictedLabel: unNormTestPredsArray[i],
    actualLabel: unNormTestLabelsArray[i]
  });
}

// Sort the sortedResults array by actualLabel
sortedResults.sort((a, b) => a.actualLabel - b.actualLabel);

// Display sorted results
for (let i = 0; i < sortedResults.length; i++) {
  const result = sortedResults[i];
  //console.log('trollolol');
  document.getElementById('results').innerText += `Predicted Label: ${result.predictedLabel}, Actual Label: ${result.actualLabel} \n`;
  if (i === 3) {
    document.getElementById('results').innerText += `\n \n \n`;
  }
}

if(Math.abs(sortedResults[3].predictedLabel - sortedResults[2].predictedLabel) < 0.3){
  document.getElementById('results').innerText += `label ${sortedResults[3].predictedLabel} and label ${sortedResults[2].predictedLabel} need to be reavaluated since they are  similiar...\n`;
  retry = true;
}

// Sort the sortedResults array by predictedLabel
sortedResults.sort((a, b) => a.predictedLabel - b.predictedLabel);

let WorkBenches = ['Terasort', 'RandomForest', 'Svd', 'World Count']; //since the data is sorted, this will work
let labelMap = {1: 'Terasort', 2: 'RandomForest', 3: 'Svd', 4: 'World Count'};

if(retry === false){
  for (let i = 0; i < sortedResults.length; i++) {
    const result = sortedResults[i];
    const predictedWorkBench = WorkBenches[i % WorkBenches.length];
    const actualWorkBench = labelMap[result.actualLabel];
  
    predictions += `Predicted WorkBench: ${predictedWorkBench}, Actual WorkBench: ${actualWorkBench} \n`;
  }
}

else{
  for (let i = 0; i < (sortedResults.length/2); i++) {
    const result = sortedResults[i];
    const predictedWorkBench = WorkBenches[i % WorkBenches.length];
    const actualWorkBench = labelMap[result.actualLabel];
  
    predictions += `Predicted WorkBench: ${predictedWorkBench}, Actual WorkBench: ${actualWorkBench} \n`;
  }
}



  const predictedPointsTest = Array.from(unNormTestPredsArray).map((val, i) => {
    return {x: test[i].sum, y: val};
  });

  const actualPointsTest = Array.from(unNormTestLabelsArray).map((val, i) => {
    return {x: test[i].sum, y: val};
  });

  tfvis.render.scatterplot(
    {name: 'Test Data Predictions vs Actual Data (Sum)'},
    {values: [actualPointsTest, predictedPointsTest], series: ['actual', 'predicted']},
    {
      xLabel: 'Sum',
      yLabel: 'Label',
      height: 300
    }
  );
}





//stop now if undo ...............................................................................................

async function reavaluate(measurementName) {
  console.log('Hello TensorFlow');

  /**
   * Get the car data reduced to just the variables we are interested
   * and cleaned of missing data.
   */
  async function getData(measurementName) {

    //data from svd3 is filtered here (the gist file doesn't have data containing svd3 since it is an outlier)
    const carsDataResponse = await fetch('https://gist.githubusercontent.com/Ralphato/c55cc499a0772d964adb2b4eec9d2820/raw/67baf6b51fd8e1369c072baa3597adc4a0c254d1/train2.JSON');
    const carsData = await carsDataResponse.json();
    const dataFiltered = carsData.filter(car => car.Measurement === measurementName);
    const cleaned = dataFiltered.map(car => ({
      sum: car.Sum,
      count: car.Count,
      label: car.Label,
    }))
    .filter(car => car.sum != null && car.count != null && car.label != null);

    // Remove outliers using IQR method
    const sums = cleaned.map(car => car.sum);
    const { lowerBound, upperBound } = getIQRBounds(sums);
    const filteredData = cleaned.filter(car => car.sum >= lowerBound && car.sum <= upperBound);

    // Filter the data to include only labels 3 and 4
    const finalFiltered = filteredData.filter(car => car.label === 3 || car.label === 4);
    return finalFiltered;
  }

  /**
  * Calculate IQR bounds for outlier detection.
  */
  function getIQRBounds(values) {
    const sorted = values.slice().sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length / 4)];
    const q3 = sorted[Math.floor(sorted.length * (3 / 4))];
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;
    return { lowerBound, upperBound };
  }

  async function run(measurementName) {
    // Load and plot the original input data that we are going to train on.
    const data = await getData(measurementName);
    const values = data.map(d => ({
      x: d.sum, // use only 1 of the 2 inputs since it is a 2d graph
      y: d.label,
    }));

    tfvis.render.scatterplot(
      {name: 'Sum v Label'},
      {values},
      {
        xLabel: 'Sum',
        yLabel: 'Label',
        height: 300
      }
    );

    // Create the model
    const model = createModel();
    tfvis.show.modelSummary({name: 'Model Summary'}, model);

    // Convert the data to a form we can use for training.
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;

    // Train the model
    await trainModel(model, inputs, labels);
    console.log('Done Training');
    // Make some predictions using the model and compare them to the
    // original data
    testModel(model, data, tensorData);
    await testRun(model, tensorData, measurementName);
  }

  function createModel() {
    // Create a sequential model
    const model = tf.sequential();

    // Add a single input layer
    model.add(tf.layers.dense({inputShape: [2], units: 50, activation: 'relu', useBias: true}));
    model.add(tf.layers.dense({units: 50, activation: 'sigmoid', useBias: true}));
    model.add(tf.layers.dense({units: 50, activation: 'sigmoid', useBias: true}));

    // Add an output layer
    model.add(tf.layers.dense({units: 1, useBias: true}));

    return model;
  }

  /**
   * Convert the input data to tensors that we can use for machine
   * learning. We will also do the important best practices of _shuffling_
   * the data and _normalizing_ the data
   * MPG on the y-axis.
   */
  function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

    return tf.tidy(() => {
      // Step 1. Shuffle the data
      tf.util.shuffle(data);

      // Step 2. Convert data to Tensor
      const inputs = data.map(d => [d.sum, d.count]);
      const labels = data.map(d => d.label);

      const inputTensor = tf.tensor2d(inputs, [inputs.length, 2]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

      // Step 3. Normalize the data to the range 0 - 1 using min-max scaling
      const inputMax = inputTensor.max(0);
      const inputMin = inputTensor.min(0);
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,
        // Return the min/max bounds so we can use them later.
        inputMax,
        inputMin,
        labelMax,
        labelMin,
      };
    });
  }

  async function trainModel(model, inputs, labels) {
    // Prepare the model for training.
    model.compile({
      optimizer: tf.train.adam(),
      loss: tf.losses.meanSquaredError,
      metrics: ['mse'],
    });

    const batchSize = 32;
    const epochs = 200;

    return await model.fit(inputs, labels, {
      batchSize,
      epochs,
      shuffle: true,
      callbacks: tfvis.show.fitCallbacks(
        { name: 'Training Performance' },
        ['loss', 'mse'],
        { height: 200, callbacks: ['onEpochEnd'] }
      )
    });
  }

  function testModel(model, inputData, normalizationData) {
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

    // Generate predictions for a range of numbers between 0 and 1 for both sum and count
    const [unNormSum, unNormCount, unNormPreds] = tf.tidy(() => {
      const sumNorm = tf.linspace(0, 1, 100);
      const countNorm = tf.linspace(0, 1, 100);

      // Combine the normalized sum and count into a 2D tensor
      const xsNorm = tf.stack([sumNorm, countNorm], 1);
      const predictions = model.predict(xsNorm);

      const unNormSum = sumNorm
        .mul(inputMax.gather(0).sub(inputMin.gather(0)))
        .add(inputMin.gather(0));

      const unNormCount = countNorm
        .mul(inputMax.gather(1).sub(inputMin.gather(1)))
        .add(inputMin.gather(1));

      const unNormPreds = predictions
        .mul(labelMax.sub(labelMin))
        .add(labelMin);

      // Un-normalize the data
      return [unNormSum.dataSync(), unNormCount.dataSync(), unNormPreds.dataSync()];
    });

    const predictedPointsSum = Array.from(unNormSum).map((val, i) => {
      return {x: val, y: unNormPreds[i]};
    });

    const originalPointsSum = inputData.map(d => ({
      x: d.sum, y: d.label,
    }));

    const predictedPointsCount = Array.from(unNormCount).map((val, i) => {
      return {x: val, y: unNormPreds[i]};
    });

    const originalPointsCount = inputData.map(d => ({
      x: d.count, y: d.label,
    }));

    tfvis.render.scatterplot(
      {name: 'Model Predictions vs Original Data (Sum)'},
      {values: [originalPointsSum, predictedPointsSum], series: ['original', 'predicted']},
      {
        xLabel: 'Sum',
        yLabel: 'Label',
        height: 300
      }
    );

    tfvis.render.scatterplot(
      {name: 'Model Predictions vs Original Data (Count)'},
      {values: [originalPointsCount, predictedPointsCount], series: ['original', 'predicted']},
      {
        xLabel: 'Count',
        yLabel: 'Label',
        height: 300
      }
    );
  }

  async function getTest(measurementName) {
    const carsDataResponse = await fetch('https://gist.githubusercontent.com/Ralphato/4e90d9b146ba86d14c91364d888f50e0/raw/798d0802d5b762a8ad61cb038c59b5f07c31dd4d/AllTestingData');
    const carsData = await carsDataResponse.json();
    const carsDataMeas = carsData.filter(car => car.Measurement === measurementName);
    const cleaned = carsDataMeas.map(car => ({
      sum: car.Sum,
      count: car.Count,
      label: car.Label,
    }))
    .filter(car => car.sum != null && car.count != null && car.label != null);
    let cleanedFiltered = cleaned.filter(car => car.label === 3 || car.label === 4);
    return cleanedFiltered;
  }

  async function testRun(model, normalizationData, measurementName) {
    const test = await getTest(measurementName);

    // Normalize the test data using the same normalization parameters as the training data
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData;

    const testInputs = test.map(d => [d.sum, d.count]);
    const testLabels = test.map(d => d.label);

    const testInputTensor = tf.tensor2d(testInputs, [testInputs.length, 2]);
    const testLabelTensor = tf.tensor2d(testLabels, [testLabels.length, 1]);

    const normalizedTestInputs = testInputTensor.sub(inputMin).div(inputMax.sub(inputMin));
    const normalizedTestLabels = testLabelTensor.sub(labelMin).div(labelMax.sub(labelMin));

    // Make predictions on the normalized test data
    const testPredictions = model.predict(normalizedTestInputs);

    // Un-normalize the predictions
    const unNormTestPreds = testPredictions
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Un-normalize the test labels for comparison
    const unNormTestLabels = normalizedTestLabels
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    // Log the predicted and actual labels for the test data
    const unNormTestPredsArray = unNormTestPreds.dataSync();
    const unNormTestLabelsArray = unNormTestLabels.dataSync();

    console.log("Sum, Count, Predicted Label, Actual Label");
    for (let i = 0; i < unNormTestPredsArray.length; i++) {
      console.log(`Sum: ${test[i].sum}, Count: ${test[i].count}, Predicted Label: ${unNormTestPredsArray[i]}, Actual Label: ${unNormTestLabelsArray[i]}`);
    }

    
  let sortedResults = [];

  // Populate the sortedResults array
  for (let i = 0; i < unNormTestPredsArray.length; i++) {
    sortedResults.push({
      predictedLabel: unNormTestPredsArray[i],
      actualLabel: unNormTestLabelsArray[i]
    });
  }
  
  // Sort the sortedResults array by actualLabel
  sortedResults.sort((a, b) => a.actualLabel - b.actualLabel);
  
  // Display sorted results
  for (let i = 0; i < sortedResults.length; i++) {
    const result = sortedResults[i];
    //console.log('trollolol');
    document.getElementById('results').innerText += `Predicted Label: ${result.predictedLabel}, Actual Label: ${result.actualLabel} \n`;
    if (i === 1) {
      document.getElementById('results').innerText += `\n \n \n`;
    }
  }
  
  // Sort the sortedResults array by predictedLabel
  sortedResults.sort((a, b) => a.predictedLabel - b.predictedLabel);
  
  let WorkBenches = ['Svd', 'World Count']; //since the data is sorted, this will work
  let labelMap = {3: 'Svd', 4: 'World Count'};
  
  for (let i = 0; i < sortedResults.length; i++) {
      const result = sortedResults[i];
      const predictedWorkBench = WorkBenches[i % WorkBenches.length];
      const actualWorkBench = labelMap[result.actualLabel];
    
      predictions += `Predicted WorkBench: ${predictedWorkBench}, Actual WorkBench: ${actualWorkBench} \n`;
  }
    

    const predictedPointsTest = Array.from(unNormTestPredsArray).map((val, i) => {
      return {x: test[i].sum, y: val};
    });

    const actualPointsTest = Array.from(unNormTestLabelsArray).map((val, i) => {
      return {x: test[i].sum, y: val};
    });

    tfvis.render.scatterplot(
      {name: 'Test Data Predictions vs Actual Data (Sum)'},
      {values: [actualPointsTest, predictedPointsTest], series: ['actual', 'predicted']},
      {
        xLabel: 'Sum',
        yLabel: 'Label',
        height: 300
      }
    );
  }

  await run(measurementName);
}
