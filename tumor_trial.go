// 02-601: Programming for Scientist
// Final Project: Construct Convolutional Neural Network From Scratch Using GOlang
// This script both training and testing function to both train and evaluate
// our simple CNN modle's performance on the "Brain Tuor modified" data set
// It's helper function of reading from input can be found in the "train.go" file
// It has a function created to initialize a "Convolution" struct, which is a single convolution layer
// This script was developed by Dunhan Jiang


package main

import (
	"fmt"
	"math"
	"math/rand"
)

// This Train function would load model and use traing set to improve the parameter of model layers.
func TrainBrainTumor(path string, learning_rate float64, num_epoch int, batch_size int) *Model {
	trainImages, err := LoadImageCSV(path + "/X_train.csv")
	if err != nil {
		panic("Load brain tumor dataset training image fail!")
	}
	fmt.Println("Brain Tumor Training Data Set Loaded:", 
				len(trainImages.Data), len(trainImages.Data[0]), len(trainImages.Data[0][0]), len(trainImages.Data[0][0][0]))
	
	trainLabels, err := LoadLabelCSV(path + "/y_train.csv")
	if err != nil {
		panic("Load brain tumor dataset training label fail!")
	}
	fmt.Println("Brain Tumor Training Data Set Label Loaded:", len(trainLabels), len(trainLabels[0]))

	//construct model
	data := []int{20, 20, 1, 2}
	kernel_1 := make([][][][]float64, data[0])
	scale := float64(math.Sqrt(float64(float64(3*data[0]*data[1]*data[2]) / float64(data[3])))) // scaler
	for i := 0; i < data[0]; i++ {
		kernel_1[i] = make([][][]float64, data[1])
		for j := 0; j < data[1]; j++ {
			kernel_1[i][j] = make([][]float64, data[2])
			for k := 0; k < data[2]; k++ {
				kernel_1[i][j][k] = make([]float64, data[3])
				for l := 0; l < data[3]; l++ {
					kernel_1[i][j][k][l] = float64(rand.NormFloat64()) / scale
				}
			}
		}
	}
	conv_1 := InitializeConvolutionLayer(kernel_1, 0, 4, batch_size)

	//pool_1
	var pool_1 Pooling

	//relu
	var relu_1 Relu

	//conv2
	data = []int{7, 7, 2, 4}
	kernel_2 := make([][][][]float64, data[0])
	scale = float64(math.Sqrt(float64(float64(3*data[0]*data[1]*data[2]) / float64(data[3])))) // scaler
	for i := 0; i < data[0]; i++ {
		kernel_2[i] = make([][][]float64, data[1])
		for j := 0; j < data[1]; j++ {
			kernel_2[i][j] = make([][]float64, data[2])
			for k := 0; k < data[2]; k++ {
				kernel_2[i][j][k] = make([]float64, data[3])
				for l := 0; l < data[3]; l++ {
					kernel_2[i][j][k][l] = float64(rand.NormFloat64()) / scale
				}
			}
		}
	}
	conv_2 := InitializeConvolutionLayer(kernel_2, 0, 1, batch_size)

	//relu_2
	var relu_2 Relu

	//pool_2
	var pool_2 Pooling

	//linear layer
	linear := NewLinear(400, 2)

	numImages := len(trainImages.Data)
	for epoch := 0; epoch < 3; epoch++ {
		for i := 0; i < numImages; i += batch_size {
			if i > numImages-batch_size {
				continue
			}
			batchData := trainImages.Data[i : i+batch_size]
			batchLabel := trainLabels[i : i+batch_size]

			//forward pass
			conv_1_output := conv_1.Forward(batchData) // 52 by 52 by 2
			relu_1.Forward(conv_1_output)
			pool_1_output := pool_1.Forward(conv_1_output) // 26 by 26 by 2
			conv_2_output := conv_2.Forward(pool_1_output) // 20 by 20 by 4
			relu_2.Forward(conv_2_output)
			pool_2_output := pool_2.Forward(conv_2_output) // 10 by 10 by 4
			pool_2_output_reshaped := Reshape4Dto2D(pool_2_output)
			linear_output := linear.Forward(pool_2_output_reshaped)
			// fmt.Println(linear_output)
			
			loss, delta := SoftmaxCalLoss(linear_output, batchLabel)
			//calculate loss
			delta = linear.Backward(delta, learning_rate)
			delta_1 := Reshape2Dto4D(delta, batch_size, 4, 10, 10)
			delta_2 := pool_2.Backward(delta_1)
			relu_2.Backward(delta_2)
			delta_3 := conv_2.Backward(delta_2, learning_rate)
			delta_4 := pool_1.Backward(delta_3)
			relu_1.Backward(delta_4)
			conv_1.Backward(delta_4, learning_rate)
			if i%30 == 0 {
				fmt.Printf("Epoch-%d-%05d : loss:%.4f\n", epoch, i, loss)
			}
		}
		learning_rate *= float64(math.Pow(0.95, float64(epoch+1)))
	}

	//return model parameters
	m := Model{
		kernel_1: conv_1.Kernel,
		kernel_2: conv_2.Kernel,
		bias_1:   conv_1.Bias,
		bias_2:   conv_2.Bias,
		weight:   linear.W,
		bias:     linear.b,
	}
	return &m	
}



func EvaluateBrainTumor(path string, batch_size int, m Model) float64 {
	//load test image
	k1 := m.kernel_1
	k2 := m.kernel_2
	b1 := m.bias_1
	b2 := m.bias_2
	w3 := m.weight
	b3 := m.bias
	//load test image
	testImages, err := LoadImageCSV(path + "/X_test.csv")
	if err != nil {
		panic("Load brain tumor dataset test image fail!")
	}

	testLabels, err := LoadLabelCSV(path + "/y_test.csv")
	if err != nil {
		panic("Load brain tumor dataset test label fail!")
	}

	//construct model
	//conv1
	conv_1 := InitializeConvolutionLayer(k1, 0, 4, batch_size)
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
	linear := NewLinear(400, 2)
	linear.W = w3
	linear.b = b3

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
		index := OneHot(softmax_output)
		for k := 0; k < batch_size; k++ {
			if batchLabel[k][index[k]] == 1 {
				correct += 1
			}
		}
	}
	Accuracy := float64(correct) / float64(numImages)
	return Accuracy
}