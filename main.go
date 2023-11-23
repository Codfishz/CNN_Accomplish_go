package main

import "fmt"

func main() {

	//training process
	fmt.Println("Start training process")

	learning_rate := float32(0.01)
	batch_size := 100
	num_epoch := 10
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
