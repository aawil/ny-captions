## “On the Internet, nobody knows you’re a bot.”

My fourth Metis project. Using the New Yorker cartoon caption contest submission data collected by NextML and archived [here](https://github.com/nextml/caption-contest-data), I used topic modeling to group about 100 cartoons into twenty broad categories, like Restaurants, Dogs, and Work. I then used Max Woolf’s [textgenrnn](https://github.com/minimaxir/textgenrnn), running on an AWS EC2 instance, to train an RNN on the caption submissions for each cartoon category. 

Finally, I created a [Flask web app](https://caption-ai.herokuapp.com/) that allows users to generate an unlimited number of new and bizarre captions for each cartoon category.
