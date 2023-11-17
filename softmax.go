package main

import (
	"math"
)

type Softmax struct {
	softmax [][]float32
}

func (s *Softmax) predict(predict [][]float32) {
	batchsize, classes := len(predict), len(predict[0])
	s.softmax = make([][]float32, batchsize)

	for i := 0; i < batchsize; i++ {
		//Initialize softmax object
		predictTmp := make([]float32, classes)

		// Find the maximum value in the array
		maxValue := predict[i][0]
		for j := 1; j < classes; j++ {
			if predict[i][j] > maxValue {
				maxValue = predict[i][j]
			}
		}

		// Calculate the softmax values given the forward propagation.
		sumExp := float32(0.0)
		for j := 0; j < classes; j++ {
			//Calculating the maximum value to maintain numerical stability
			predictTmp[j] = float32(math.Exp(float64(predict[i][j] - maxValue)))
			sumExp += predictTmp[j]
		}

		for j := 0; j < classes; j++ {
			predictTmp[j] /= sumExp
		}

		s.softmax[i] = predictTmp
	}
	return &s
}

func (s *Softmax) CalLoss(predict [][]float32, label [][]float32) (float32, [][]float32) {
	batchsize, classes := len(predict), len(predict[0])
	// Calculate the softmax values

	var softmax Softmax
	softmax = s.predict(predict)

	loss := float32(0.0)
	//Initialize delta matrix
	delta := make([][]float32, batchsize)

	for i := 0; i < batchsize; i++ {
		delta[i] = make([]float32, classes)
		for j := 0; j < classes; j++ {
			delta[i][j] = softmax.softmax[i][j] - label[i][j]
			loss -= float32(math.Log(float64(softmaxs.softmax[i][j]))) * label[i][j]
		}
	}

	loss /= float32(batchsize*classes)
	return loss, delta
}