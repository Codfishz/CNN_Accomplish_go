package main

import (
	"math"
	"math/rand"
	// "fmt"
	// "time"
)

// Linear represents a fully connected neural network layer.
type Linear struct {
	W           [][]float64 // Weight matrix, inChannel x outChannel
	b           []float64	// Bias vector, outChannel
	WGradient   [][]float64 // Weight gradient matrix
	bGradient   []float64	// Bias gradient vector
	x           [][]float64 // Input data
	inChannel   int			// Number of input features
	outChannel  int			// Number of output features
}

// NewLinear creates a new Linear layer with the specified input and output sizes.
func NewLinear(inChannel, outChannel int) *Linear {
	scale := math.Sqrt(float64(inChannel) / 2)

	// Initialize the weights and biases with random values (gradients will be zero)
	W := make([][]float64, inChannel)
	WGradient := make([][]float64, inChannel)
	for i := range W {
		W[i] = make([]float64, outChannel)
		WGradient[i] = make([]float64, outChannel)
		for j := range W[i] {
			W[i][j] = float64(rand.NormFloat64() / scale)
			// W[i][j] = float64(0.1)
		}
	}

	b := make([]float64, outChannel)
	bGradient := make([]float64, outChannel)
	for i := range b {
		b[i] = float64(rand.NormFloat64() / scale)
		// b[i] = float64(0.1)
		bGradient[i] = 0
	}

	// Return the new layer
	return &Linear{
		W:           W,
		b:           b,
		WGradient:   WGradient,
		bGradient:   bGradient,
		inChannel:   inChannel,
		outChannel:  outChannel,
	}
}

// Forward computes the forward pass of the linear layer.
func (l *Linear) Forward(x [][]float64) [][]float64 {
	// copy x to l.x
	l.x = make([][]float64, len(x))
	for i := range x {
		l.x[i] = make([]float64, len(x[i]))
		for j := range x[i] {
			l.x[i][j] = x[i][j]
		}
	}


	xForward := make([][]float64, len(x)) // next layer input

	// xForward = x * W + b
	for i := range xForward {
		xForward[i] = make([]float64, l.outChannel)
		for j := range xForward[i] {
			sum := l.b[j]
			for k := range x[i] {
				sum += x[i][k] * l.W[k][j]
			}
			xForward[i][j] = sum
		}
	}

	return xForward
}

// Backward computes the backward pass of the linear layer.
func (l *Linear) Backward(delta [][]float64, learningRate float64) [][]float64 {
	batchSize := len(delta)
	deltaBackward := make([][]float64, batchSize) // previous layer delta

	for i := range l.WGradient {
		for j := range l.WGradient[i] {
			l.WGradient[i][j] = float64(0.0)
			l.bGradient[j] = float64(0.0)
		}
	}

	// wGradient = x^T * delta / batchSize
	// deltaBackward = delta * w^T
	batchSizeF32 := float64(batchSize)
	// fmt.Println(len(l.WGradient), len(l.WGradient[0]))
	for i := range deltaBackward { // 100
		deltaBackward[i] = make([]float64, l.inChannel)
		for j := range deltaBackward[i] { // 256
			for k := range delta[i] { // 10
				l.WGradient[j][k] += l.x[i][j] * delta[i][k] / batchSizeF32
				deltaBackward[i][j] += delta[i][k] * l.W[j][k]
			}
			// l.WGradient[j][i] /= float64(batchSize)
		}
	}

	// bGradient = delta / batchSize
	for i := range l.b {
		for j := range delta {
			l.bGradient[i] += delta[j][i]  / batchSizeF32
		}
		// l.bGradient[i] /= float64(batchSize)
	}

	// Backpropagation
	// w = w - wGradient * learningRate
	for i := range l.W {
		for j := range l.W[i] {
			l.W[i][j] -= l.WGradient[i][j] * learningRate
		}
	}

	// b = b - bGradient * learningRate
	for i := range l.b {
		l.b[i] -= l.bGradient[i] * learningRate
	}

	return deltaBackward
}
