package main

func eval(path string, batch_size int, k1, k2 [][][][]float32, b1, b2 []float32, w3, b3 [][]float32) float32 {
	//load test image
	testImages, err := LoadImagesFromFile(path + "/t10k-images-idx3-ubyte")
	if err != nil {
		panic("Load test image fail!")
	}

	testLabels, err := LoadLabelsFromFile(path + "/t10k-labels-idx1-ubyte")
	if err != nil {
		panic("Load test label fail!")
	}

	//construct model
	//conv1
	conv_1 := InitializeConvolutionLayer(k1, 0, 1, batch_size)

	//pool_1
	var pool_1 Pooling

	//relu
	var relu_1 Relu

	//conv2
	conv_2 := InitializeConvolutionLayer(k2, 0, 1, batch_size)

	//relu_2
	var relu_2 Relu

	//pool_2
	var pool_2 Pooling

	//linear layer
	linear := NewLinear(256, 10)

	//softmax
	var softmax Softmax

	//evaluation
	correct := 0
	numImages := len(testImages.Data)
	var index int
	for i := 0; i < numImages; i += batch_size {
		//get batch data
		batchData := testImages.Data[i : i+batch_size]
		batchLabel := testLabels[i : i+batch_size]

		//forward pass
		conv_1_output := conv_1.Forward(batchData)
		pool_1_output := pool_1.Forward(conv_1_output)
		relu_1.Forward(pool_1_output)
		conv_2_output := conv_2.Forward(pool_1_output)
		relu_2.Forward(conv_2_output)
		pool_2_output := pool_2.Forward(conv_2_output)
		linear_output := linear.Forward(pool_2_output)

		softmax_output := softmax.predict(linear_output)

		index = OneHot(softmax_output.softmax)
		if batchLabel[i][index] == 1 {
			correct++
		}
	}
	Accuracy := float32(correct / numImages)

	return Accuracy
}

func OneHot(predictedLabel [][]float32) int {
	var max float32
	var index int
	for i := 0; i < len(predictedLabel[0]); i++ {
		if predictedLabel[0][i] > max {
			index = i
			max = predictedLabel[0][i]
		}
	}
	return index
}
