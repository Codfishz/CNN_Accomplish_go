package main

type Tensor struct {
	Data     [][]float32
	Shape    []int
	DataType string
	Pad int
	Stride int
}

func (tensor Tensor) forward(kernalShape []int) Tensor {

	return tensor
}

func (tensor Tensor) backward() Tensor {

	return tensor
}