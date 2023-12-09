// 02-601: Programming for Scientist
// Final Project: Construct Convolutional Neural Network From Scratch Using GOlang
// This script both training and testing function to both train and evaluate
// our simple CNN modle's performance on the "Brain Tuor modified" data set
// It has a function created to initialize a "Convolution" struct, which is a single convolution layer
// This script was developed by Dunhan Jiang (Train & Test) and Zirui Chen (LoadImage & LoadLabel)

package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"math/rand"
	"os"
	"strconv"
)

// This Train function would load model and use traing set to improve the parameter of model layers.
func TrainBrainTumor(path string, learning_rate float64, num_epoch int, batch_size int) *Model_Mutiple {
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
	data := []int{5, 5, 1, 4}
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
	conv_1 := InitializeConvolutionLayer(kernel_1, 0, 1, batch_size)

	//pool_1
	var pool_1 Pooling

	//relu
	var relu_1 Relu

	//conv2
	data = []int{5, 5, 4, 8}
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

	//conv3
	data = []int{6, 6, 8, 12}
	kernel_3 := make([][][][]float64, data[0])
	scale = float64(math.Sqrt(float64(float64(3*data[0]*data[1]*data[2]) / float64(data[3])))) // scaler
	for i := 0; i < data[0]; i++ {
		kernel_3[i] = make([][][]float64, data[1])
		for j := 0; j < data[1]; j++ {
			kernel_3[i][j] = make([][]float64, data[2])
			for k := 0; k < data[2]; k++ {
				kernel_3[i][j][k] = make([]float64, data[3])
				for l := 0; l < data[3]; l++ {
					kernel_3[i][j][k][l] = float64(rand.NormFloat64()) / scale
				}
			}
		}
	}
	conv_3 := InitializeConvolutionLayer(kernel_3, 0, 1, batch_size)

	//relu_2
	var relu_3 Relu

	//pool_2
	var pool_3 Pooling

	//conv4
	data = []int{5, 5, 12, 16}
	kernel_4 := make([][][][]float64, data[0])
	scale = float64(math.Sqrt(float64(float64(3*data[0]*data[1]*data[2]) / float64(data[3])))) // scaler
	for i := 0; i < data[0]; i++ {
		kernel_4[i] = make([][][]float64, data[1])
		for j := 0; j < data[1]; j++ {
			kernel_4[i][j] = make([][]float64, data[2])
			for k := 0; k < data[2]; k++ {
				kernel_4[i][j][k] = make([]float64, data[3])
				for l := 0; l < data[3]; l++ {
					kernel_4[i][j][k][l] = float64(rand.NormFloat64()) / scale
				}
			}
		}
	}
	conv_4 := InitializeConvolutionLayer(kernel_4, 0, 1, batch_size)

	//relu_2
	var relu_4 Relu

	//pool_2
	var pool_4 Pooling

	//linear layer
	linear := NewLinear(10*10*16, 2)

	numImages := len(trainImages.Data)
	for epoch := 0; epoch < 3; epoch++ {
		for i := 0; i < numImages; i += batch_size {
			if i > numImages-batch_size {
				continue
			}
			batchData := trainImages.Data[i : i+batch_size]
			batchLabel := trainLabels[i : i+batch_size]

			//forward pass
			conv_1_output := conv_1.Forward(batchData) // 220 by 220 by 4
			relu_1.Forward(conv_1_output)
			pool_1_output := pool_1.Forward(conv_1_output) // 110 by 110 by 4
			conv_2_output := conv_2.Forward(pool_1_output) // 106 by 106 by 8
			relu_2.Forward(conv_2_output)
			pool_2_output := pool_2.Forward(conv_2_output) // 53 by 53 by 6
			conv_3_output := conv_3.Forward(pool_2_output) // 48 by 48 by 12
			relu_3.Forward(conv_3_output)
			pool_3_output := pool_3.Forward(conv_3_output) // 24 by 24 by 12
			conv_4_output := conv_4.Forward(pool_3_output) // 20 by 20 by 16
			relu_4.Forward(conv_4_output)
			pool_4_output := pool_4.Forward(conv_4_output) // 10 by 10 by 16

			pool_4_output_reshaped := Reshape4Dto2D(pool_4_output)

			linear_output := linear.Forward(pool_4_output_reshaped)
			// fmt.Println(linear_output)

			loss, delta := SoftmaxCalLoss(linear_output, batchLabel)
			//calculate loss
			delta = linear.Backward(delta, learning_rate)
			// fmt.Println("delta")
			// fmt.Println(len(delta))
			// fmt.Println(len(delta[0]))
			delta_1 := Reshape2Dto4D(delta, batch_size, 16, 10, 10)
			// fmt.Println("delta_1")
			// fmt.Println(len(delta_1))
			// fmt.Println(len(delta_1[0]))
			// fmt.Println(len(delta_1[0][0]))
			// fmt.Println(len(delta_1[0][0][0]))
			delta_2 := pool_4.Backward(delta_1)
			relu_4.Backward(delta_2)
			// fmt.Println("delta_2")
			// fmt.Println(len(delta_2))
			// fmt.Println(len(delta_2[0]))
			// fmt.Println(len(delta_2[0][0]))
			// fmt.Println(len(delta_2[0][0][0]))
			delta_3 := conv_4.Backward(delta_2, learning_rate)
			// fmt.Println("delta_3")
			// fmt.Println(len(delta_3))
			// fmt.Println(len(delta_3[0]))
			// fmt.Println(len(delta_3[0][0]))
			// fmt.Println(len(delta_3[0][0][0]))
			delta_4 := pool_3.Backward(delta_3)
			// fmt.Println("delta_4")
			// fmt.Println(len(delta_4))
			// fmt.Println(len(delta_4[0]))
			// fmt.Println(len(delta_4[0][0]))
			// fmt.Println(len(delta_4[0][0][0]))
			relu_3.Backward(delta_4)
			delta_5 := conv_3.Backward(delta_4, learning_rate)

			delta_6 := pool_2.Backward(delta_5)

			relu_2.Backward(delta_6)

			delta_7 := conv_2.Backward(delta_6, learning_rate)

			delta_8 := pool_1.Backward(delta_7)

			relu_1.Backward(delta_8)

			conv_1.Backward(delta_8, learning_rate)

			if i%30 == 0 {
				fmt.Printf("Epoch-%d-%05d : loss:%.4f\n", epoch, i, loss)
			}
		}
		learning_rate *= float64(math.Pow(0.95, float64(epoch+1)))
	}

	//return model parameters
	m := Model_Mutiple{
		kernel_1: conv_1.Kernel,
		kernel_2: conv_2.Kernel,
		kernel_3: conv_3.Kernel,
		kernel_4: conv_4.Kernel,
		bias_1:   conv_1.Bias,
		bias_2:   conv_2.Bias,
		bias_3:   conv_3.Bias,
		bias_4:   conv_4.Bias,
		weight:   linear.W,
		bias:     linear.b,
	}
	return &m
}

func EvaluateBrainTumor(path string, batch_size int, m Model_Mutiple) float64 {
	//load test image
	k1 := m.kernel_1
	k2 := m.kernel_2
	k3 := m.kernel_3
	k4 := m.kernel_4
	b1 := m.bias_1
	b2 := m.bias_2
	b3 := m.bias_3
	b4 := m.bias_4
	w := m.weight
	b := m.bias
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

	//conv3
	conv_3 := InitializeConvolutionLayer(k3, 0, 1, batch_size)
	conv_3.Bias = b3

	//relu_2
	var relu_3 Relu

	//pool_2
	var pool_3 Pooling

	//conv4
	conv_4 := InitializeConvolutionLayer(k4, 0, 1, batch_size)
	conv_4.Bias = b4

	//relu_2
	var relu_4 Relu

	//pool_2
	var pool_4 Pooling

	//linear layer
	linear := NewLinear(10*10*16, 2)
	linear.W = w
	linear.b = b

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
		conv_1_output := conv_1.Forward(batchData) // 220 by 220 by 4
		relu_1.Forward(conv_1_output)
		pool_1_output := pool_1.Forward(conv_1_output) // 110 by 110 by 4
		conv_2_output := conv_2.Forward(pool_1_output) // 106 by 106 by 8
		relu_2.Forward(conv_2_output)
		pool_2_output := pool_2.Forward(conv_2_output) // 53 by 53 by 6
		conv_3_output := conv_3.Forward(pool_2_output) // 48 by 48 by 12
		relu_3.Forward(conv_3_output)
		pool_3_output := pool_3.Forward(conv_3_output) // 24 by 24 by 12
		conv_4_output := conv_4.Forward(pool_3_output) // 20 by 20 by 16
		relu_4.Forward(conv_4_output)
		pool_4_output := pool_4.Forward(conv_4_output) // 10 by 10 by 16

		pool_4_output_reshaped := Reshape4Dto2D(pool_4_output)

		linear_output := linear.Forward(pool_4_output_reshaped)

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

// LoadImageCSV reads flattened pixel values from a CSV file and returns a 4D tensor.
// The tensor size is [number of images][channel of image][image height][image width].
func LoadImageCSV(imageFile string) (*Tensor, error) {
	file, err := os.Open(imageFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	numImages := len(records)
	numPixels := len(records[0])
	numRows := math.Sqrt(float64(numPixels))
	numCols := numRows

	imageData := make([][][][]float64, numImages)
	for i := 0; i < numImages; i++ {
		imageData[i] = make([][][]float64, 1)
		imageData[i][0] = make([][]float64, int(numRows))

		for r := 0; r < int(numRows); r++ {
			imageData[i][0][r] = make([]float64, int(numCols))
			for c := 0; c < int(numCols); c++ {
				pixelValue, err := strconv.ParseFloat(records[i][r*int(numCols)+c], 64)
				if err != nil {
					return nil, err
				}
				imageData[i][0][r][c] = pixelValue
			}
		}
	}

	return &Tensor{Data: imageData}, nil
}

// LoadLabelCSV reads one-hot encoded labels from a CSV file and returns a 2D tensor.
// The tensor size is [number of labels][number of classes].
func LoadLabelCSV(labelFile string) ([][]float64, error) {
	file, err := os.Open(labelFile)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	numLabels := len(records[0])
	fmt.Println(numLabels)
	labelData := make([][]float64, numLabels)
	for i := 0; i < numLabels; i++ {
		labelData[i] = make([]float64, 2)

		labelVal, err := strconv.Atoi(records[0][i])
		if err != nil {
			return nil, err
		}
		if labelVal == 0 {
			labelData[i][0] = 1
			labelData[i][1] = 0
		} else {
			labelData[i][0] = 0
			labelData[i][1] = 1
		}
	}

	return labelData, nil
}
