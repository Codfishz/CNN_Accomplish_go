package main

import (
	"math"
	// "fmt"
)


func SoftmaxPredict(predict [][]float64) [][]float64 {
	batchsize, classes := len(predict), len(predict[0])
	softmax := make([][]float64, batchsize)

	for i := 0; i < batchsize; i++ {
		//Initialize softmax object
		predictTmp := make([]float64, classes)

		// Find the maximum value in the array
		maxValue := predict[i][0]
		for j := 1; j < classes; j++ {
			if predict[i][j] > maxValue {
				maxValue = predict[i][j]
			}
		}
		// fmt.Println("maximum:", maxValue)

		// Calculate the softmax values given the forward propagation.
		sumExp := float64(0.0)
		for j := 0; j < classes; j++ {
			//Calculating the maximum value to maintain numerical stability
			predictTmp[j] = float64(math.Exp(float64(predict[i][j] - maxValue)))
			sumExp += predictTmp[j]
		}

		for j := 0; j < classes; j++ {
			predictTmp[j] /= sumExp
		}

		softmax[i] = predictTmp
		// fmt.Println(predictTmp)
	}
	return softmax
}

func SoftmaxCalLoss(predict [][]float64, label [][]float64) (float64, [][]float64) {
	batchsize, classes := len(predict), len(predict[0])
	// Calculate the softmax values

	softmax := SoftmaxPredict(predict)
	// fmt.Println("predict:")
	// fmt.Println(softmax[0])
	// fmt.Println(softmax[1])
	// fmt.Println(softmax[2])
	loss := float64(0.0)
	//Initialize delta matrix
	delta := make([][]float64, batchsize)

	for i := 0; i < batchsize; i++ {
		delta[i] = make([]float64, classes)
		for j := 0; j < classes; j++ {
			delta[i][j] = softmax[i][j] - label[i][j]
			// fmt.Println("delta:", delta[i][j])
			loss -= float64(math.Log(float64(softmax[i][j]))) * label[i][j]
			// fmt.Println(float64(math.Log(float64(softmax[i][j]))))
		}
	}
	// fmt.Println("loss adds up:", loss)
	loss /= float64(batchsize)
	return loss, delta
}
