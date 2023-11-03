package main

type Tensor struct {
	//first dimension for batch size
	//second dimentsion for channel number
	//third and fourth dimentsion for width and height
	Data [][][][]float32
}
