let nn;
let l; // label

// data for XOR
// [1, 0] -> [1]
// [0, 1] -> [1]
// [1, 1] -> [0]
// [0, 0] -> [0]
let training_data = [
    {
        inputs: [0, 1],
        targets: [1]
    },
    {
        inputs: [1, 0],
        targets: [1]
    },
    {
        inputs: [0, 0],
        targets: [0]
    },
    {
        inputs: [1, 1],
        targets: [0]
    },
];

function setup() {
    createCanvas(400, 400);
    nn = new NeuralNetwork(2, 32, 1);
}

function draw() {
    background(0);

    // training
    for (let i = 0; i < 1000; i++) {
        let data = random(training_data);
        nn.train(data.inputs, data.targets);
    }
    
    let resolution = 10;
    let cols = width / resolution;
    let rows= height / resolution;
    for (let i = 0; i < cols; i++) {
        for (let j = 0; j < rows; j++) {
            let x1 = i / cols;
            let x2 = j / rows;
            let inputs = [x1, x2];
            let y = nn.feedforward(inputs);
            fill(y * 255);
            noStroke();
            rect(i * resolution, j* resolution, resolution, resolution);
        }
    }

    // labels
    l = '[0, 0]';
    textSize(16);
    fill(255, 0, 0);
    stroke(255);
    text(l, 0, 16);

    l = '[1, 0]';
    textSize(16);
    fill(255, 0, 0);
    stroke(255);
    text(l, width - 36, 16);

    l = '[1, 1]';
    textSize(16);
    fill(255, 0, 0);
    stroke(255);
    text(l, width - 36, height - 8);

    l = '[0, 1]';
    textSize(16);
    fill(255, 0, 0);
    stroke(255);
    text(l, 0, height - 8);

}