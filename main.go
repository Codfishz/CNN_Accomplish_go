package main

import "fmt"

func main() {

	//training process
	fmt.Println("Start training process")

	learning_rate := 0.01
	batch_size := 100
	num_epoch := 10
	path := "data"
	k1, b1, k2, b2, w3, b3 = Train(path, learning_rate, num_epoch, batch_size)

	//finish training
	fmt.Println("End training process")

	//evaluation process
	fmt.Println("Start evaluation process")
	Accuracy := Eval(path, k1, b1, k2, b2, w3, b3)
	fmt.Println("Accuracy is ", Accuracy)

	fmt.Println("End training process")

}
