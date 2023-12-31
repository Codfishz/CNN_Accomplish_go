package main

import (
	// "fmt"
)

func Eval(path string, batch_size int, m Model) float64 {
	//load test image
	k1 := m.kernel_1
	k2 := m.kernel_2
	b1 := m.bias_1
	b2 := m.bias_2
	w3 := m.weight
	b3 := m.bias
	//load test image
	testImages, err := LoadImagesFromFile(path + "/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte")
	if err != nil {
		panic("Load test image fail!")
	}

	testLabels, err := LoadLabelsFromFile(path + "/t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte")
	if err != nil {
		panic("Load test label fail!")
	}

	//construct model
	//conv1
	conv_1 := InitializeConvolutionLayer(k1, 0, 1, batch_size)
	conv_1.Bias = b1

	//pool_1
	var pool_1 Pooling

	//relu
	var relu_1 Relu

	//conv2
	conv_2 := InitializeConvolutionLayer(k2, 0, 1, batch_size)
	conv_2.Bias = b2
	//relu_2
	var relu_2 Relu

	//pool_2
	var pool_2 Pooling

	//linear layer
	linear := NewLinear(256, 10)
	linear.W = Copy2D(w3)
	linear.b = Copy1D(b3)

	//evaluation
	correct := 0
	numImages := len(testImages.Data)
	for i := 0; i < numImages; i += batch_size {
		// for i := 0; i < 300; i += batch_size {
		if i > numImages-batch_size {
			continue
		}
		//get batch data
		batchData := testImages.Data[i : i+batch_size]
		batchLabel := testLabels[i : i+batch_size]

		//forward pass
		conv_1_output := conv_1.Forward(batchData)
		relu_1.Forward(conv_1_output)
		pool_1_output := pool_1.Forward(conv_1_output)
		conv_2_output := conv_2.Forward(pool_1_output)
		relu_2.Forward(conv_2_output)
		pool_2_output := pool_2.Forward(conv_2_output)
		pool_2_output_reshaped := Reshape4Dto2D(pool_2_output)
		linear_output := linear.Forward(pool_2_output_reshaped)

		softmax_output := SoftmaxPredict(linear_output)
		//fmt.Println(softmax_output)
		index := OneHot(softmax_output)
		// fmt.Println(batchData[0])
		// fmt.Println(softmax_output.softmax)
		//fmt.Println(batchLabel)
		// fmt.Println(index)
		// fmt.Println(batchLabel[0])
		// fmt.Println(batchLabel[1])
		// fmt.Println(batchLabel[2])
		//fmt.Println("")
		for k := 0; k < batch_size; k++ {
			if batchLabel[k][index[k]] == 1 {
				correct += 1
			}
		}
	}
	Accuracy := float64(correct) / float64(numImages)
	return Accuracy
}

func OneHot(predictedLabel [][]float64) []int {
	index := make([]int, len(predictedLabel))
	for i := 0; i < len(predictedLabel); i++ {
		var max float64
		for j := 0; j < len(predictedLabel[0]); j++ {
			if predictedLabel[i][j] > max {
				index[i] = j
				max = predictedLabel[i][j]
			}
		}
	}
	return index
}
