package main

import (
	"math"
	"math/rand"
	// "time"
)

// Linear represents a fully connected neural network layer.
type Linear struct {
	W           [][]float32 // Weight matrix, inChannel x outChannel
	b           []float32	// Bias vector, outChannel, added to each row of the output
	WGradient   [][]float32 // Weight gradient matrix
	bGradient   []float32	// Bias gradient vector
	x           [][]float32 // Input data, flattened
	inChannel   int			// Number of input features
	outChannel  int			// Number of output features
}

// NewLinear creates a new Linear layer with the specified input and output sizes.
func NewLinear(inChannel, outChannel int) *Linear {
	scale := math.Sqrt(float64(inChannel) / 2)

	// Initialize the weights and biases with random values (gradients will be zero)
	W := make([][]float32, inChannel)
	WGradient := make([][]float32, inChannel)
	for i := range W {
		W[i] = make([]float32, outChannel)
		WGradient[i] = make([]float32, outChannel)
		for j := range W[i] {
			W[i][j] = float32(rand.NormFloat64() / scale)
		}
	}

	b := make([]float32, outChannel)
	bGradient := make([]float32, outChannel)
	for i := range b {
		b[i] = float32(rand.NormFloat64() / scale)
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
func (l *Linear) Forward(x [][]float32) [][]float32 {
	l.x = x
	xForward := make([][]float32, len(x)) // next layer input

	// xForward = x * W + b
	for i := range xForward {
		xForward[i] = make([]float32, l.outChannel)
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
func (l *Linear) Backward(delta [][]float32, learningRate float32) [][]float32 {
	batchSize := len(delta)
	deltaBackward := make([][]float32, batchSize) // previous layer delta

	// wGradient = x^T * delta / batchSize
	// deltaBackward = delta * w^T
	for i := range deltaBackward {
		deltaBackward[i] = make([]float32, l.inChannel)
		for j := range deltaBackward[i] {
			for k := range delta[i] {
				l.WGradient[j][k] += l.x[i][j] * delta[i][k]
				deltaBackward[i][j] += delta[i][k] * l.W[j][k]
			}
			l.WGradient[j][i] /= float32(batchSize)
		}
	}

	// bGradient = delta / batchSize
	for i := range l.b {
		for j := range delta {
			l.bGradient[i] += delta[j][i]
		}
		l.bGradient[i] /= float32(batchSize)
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

// import (
// 	"fmt"
// 	"time"
// 	"math/rand"
// )

// func main() {
// 	// Example usage:
// 	inChannels := 784  // For example, for an image of 28x28 pixels
// 	outChannels := 10  // For example, for 10 classes (0-9)
// 	layer := NewLinear(inChannels, outChannels)

// 	// Example input (batch of images)
// 	x := make([][]float32, 2) // Batch size of 2
// 	for i := range x {
// 		x[i] = make([]float32, inChannels)
// 		// Fill x[i]
// 		for j := range x[i] {
// 			x[i][j] = rand.Float32()
// 		}
// 	}

// 	fmt.Println("Input:")
// 	for i := range x {
// 		fmt.Println("Image", i+1)
// 		printData(x[i])
// 	}


// 	// Forward pass
// 	output := layer.Forward(x)

// 	fmt.Println("\nOutput:")
// 	for i := range output {
// 		fmt.Println("Image", i+1)
// 		printData(output[i])
// 	}

// 	// Assume some gradient coming back from the next layer
// 	delta := make([][]float32, 2) // Batch size of 2
// 	for i := range delta {
// 		delta[i] = make([]float32, outChannels)
// 		// Fill delta[i]
// 		for j := range delta[i] {
// 			delta[i][j] = rand.Float32()
// 		}
// 	}

// 	// Backward pass
// 	layer.Backward(delta, 0.01) // Example learning rate

// }

// // printData prints the data in a 2D slice
// func printData(data []float32) {
// 	for i := 0; i < len(data); i++ {
// 		fmt.Printf("%.4f ", data[i])
// 	}
// 	fmt.Println()
// }