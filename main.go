package main

import "fmt"

func main() {

	//training process
	fmt.Println("Start training process")

	learning_rate := float32(0.95)
	batch_size := 3
	num_epoch := 1
	path := "MINST"

	m := Train(path, learning_rate, num_epoch, batch_size)

	//finish training
	fmt.Println("End training process")

	//evaluation process
	fmt.Println("Start evaluation process")
	Accuracy := Eval(path, batch_size, *m)
	fmt.Println("Accuracy is ", Accuracy)

	fmt.Println("End training process")

}
