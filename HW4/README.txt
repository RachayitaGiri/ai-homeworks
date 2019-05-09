CECS 551 - Homework 4 - Rachayita Giri (017524687)

-----------
Question 1)
-----------

Training the model on 6000 training samples and 180 validation samples, I get the following results:
	- Epochs = 1 
		-- Training Loss: 	0.5797
		-- Training Accuracy: 	69.02 %
		-- Validation Loss: 	0.4554
		-- Validation Accuracy: 85.80 %
	- Epochs = 2
		-- Training Loss: 	0.4426
                -- Training Accuracy: 	78.80 %
                -- Validation Loss: 	0.3994
                -- Validation Accuracy: 83.52 %

I believe that I get a sufficiently high accuracy because of the following two reasons:
	- the pretrained VGG network has highly optimized weights that yield good results even after we add our additional layers 
	- dataset is very well defined, in the sense that the images are not very varied in terms of the features. Both, cats and dogs are very prominent objects in the context of their images.   

-----------
Question 2)
-----------

Training the model on 30,000 training samples and 900 validation samples, I get the following results:
	- Epochs = 2 
		-- Training Loss: 	0.2991
		-- Training Accuracy: 	86.49 %
		-- Validation Loss: 	0.2293
		-- Validation Accuracy: 89.84 %

Here, I can see a better accuracy because:
	- the model had more time to train in the case of 2 epochs
	- there was more data to learn from

So if I would let my transfer model train for around 5 epochs and even more, I should expect to see some more improvement in the accuracy, at least till some time.

