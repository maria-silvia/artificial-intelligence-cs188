import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"

        # compute the dot product of the stored weight vector and the given input
        return nn.DotProduct(self.w, x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"

        dot_product_node = self.run(x)
        dot_product_number = nn.as_scalar(dot_product_node)
        
        if dot_product_number >= 0:
            return 1  
        else:
            return -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        
        # repeatedly loop over the dataset 
        misclassifications = True
        while misclassifications:

            # until an entire pass over the data set 
            # is completed without making any misclassifications
            misclassifications = False

            # for each training instance
            for x, y in dataset.iterate_once(1):

                # classify with current weights
                predicted_class = self.get_prediction(x)

                # if misclassified: adjust the weight vector
                correct_class = nn.as_scalar(y)
                if predicted_class != correct_class:
                    misclassifications = True
                    self.w.update(x, correct_class)

        # 100% training accuracy has been achieved, and training can terminate.

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # recommended hyperparameters by enunciado
        hidden_layer_size = 512
        self.batch_size = 200
        self.learning_rate = 0.05

        # create trainable parameters   
        self.W1 = nn.Parameter(1, hidden_layer_size) 
        self.b1 = nn.Parameter(1, hidden_layer_size)

        self.W2 = nn.Parameter(hidden_layer_size, 1)
        self.b2 = nn.Parameter(1, 1)


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        # compute predictions for y with 2 layers
        xW1 = nn.Linear(x, self.W1)
        hidden_layer_1 = nn.ReLU(nn.AddBias(xW1, self.b1))
        
        h1W2 = nn.Linear(hidden_layer_1, self.W2)
        hidden_layer_2 = nn.AddBias(h1W2, self.b2)
        
        return hidden_layer_2 # y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        # repeatedly perform parameters updates
        # until loss is minimized (less than 0.02)
        loss = 1
        while loss > 0.02:
            
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                
                parameters = [self.W1, self.W2, self.b1, self.b2]
                gradients = nn.gradients(loss, parameters)
                grad_wrt_W1, grad_wrt_W2, grad_wrt_b1, grad_wrt_b2 = gradients
                
                self.W1.update(grad_wrt_W1, -self.learning_rate)
                self.W2.update(grad_wrt_W2, -self.learning_rate)
                self.b1.update(grad_wrt_b1, -self.learning_rate)
                self.b2.update(grad_wrt_b2, -self.learning_rate)
            
            new_loss = self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))
            loss = nn.as_scalar(new_loss)
            # print(loss)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # recommended hyperparameters by enunciado
        hidden_layer_size = 200
        self.batch_size = 100

        # create trainable parameters   
        self.W1 = nn.Parameter(784, hidden_layer_size) 
        self.b1 = nn.Parameter(1, hidden_layer_size)

        self.W2 = nn.Parameter(hidden_layer_size, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # compute predictions for y with 2 layers
        xW1 = nn.Linear(x, self.W1)
        hidden_layer_1 = nn.ReLU(nn.AddBias(xW1, self.b1))
        
        h1W2 = nn.Linear(hidden_layer_1, self.W2)
        hidden_layer_2 = nn.AddBias(h1W2, self.b2)
        
        return hidden_layer_2 # y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # repeatedly perform parameters updates
        # until accuracy is great enough so at testing it would be at least 97%
        decreasing_learning_rate = 0.8
        min_learning_rate = 0.5

        accuracy = 0
        while accuracy < 0.975:
            self.learning_rate = max(min_learning_rate, decreasing_learning_rate)
            print("learningn rate", self.learning_rate)
            
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                
                parameters = [self.W1, self.W2, self.b1, self.b2]
                gradients = nn.gradients(loss, parameters)
                grad_wrt_W1, grad_wrt_W2, grad_wrt_b1, grad_wrt_b2 = gradients
                
                self.W1.update(grad_wrt_W1, -self.learning_rate)
                self.W2.update(grad_wrt_W2, -self.learning_rate)
                self.b1.update(grad_wrt_b1, -self.learning_rate)
                self.b2.update(grad_wrt_b2, -self.learning_rate)
            
            decreasing_learning_rate -= 0.05
            accuracy = dataset.get_validation_accuracy()
            print("accuracy", accuracy)


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        
        # hyperparameters 
        hidden_layer_size = 200 # should be sufficiently large
        self.batch_size = 100
        self.learning_rate = 0.1


        # entrada tem o tamanho da quantidade caracteres unicos
        self.W1 = nn.Parameter(self.num_chars, hidden_layer_size)
        self.b1 = nn.Parameter(1, hidden_layer_size)

        self.W2 = nn.Parameter(hidden_layer_size, hidden_layer_size)
        self.b2 = nn.Parameter(1, hidden_layer_size)
        
        self.W3 = nn.Parameter(hidden_layer_size, len(self.languages))
        self.b3 = nn.Parameter(1, len(self.languages))
        # e saida tem o tamanho da quantidade de linguagens possiveis

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # primeiro caracter inicializa f como em modelos anteriores
        x0W1 = nn.Linear(xs[0], self.W1)
        hidden_layer_1 = nn.ReLU(nn.AddBias(x0W1, self.b1))

        f_initial = hidden_layer_1

        # z0
        zi = nn.ReLU(nn.AddBias(f_initial, self.b1)) 

        # combinar resultado com os proximos caracteres
        for xi in xs[1:]:

            # apply sub-network f to generate next hidden layer
            # hi = f(h_anterior, letter)
            ziW2 = nn.Linear(zi, self.W2)
            ziW2_b2 = nn.ReLU(nn.AddBias(ziW2, self.b2))

            xiW1 = nn.Linear(xi, self.W1)
            xiW1_b1 = nn.ReLU(nn.AddBias(xiW1, self.b1))

            zi = nn.Add(ziW2_b2, xiW1_b1)

        # ultimo layer, scores das palavras
        f_final = nn.AddBias(nn.Linear(zi, self.W3), self.b3) 
        return f_final


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        # repeatedly perform parameters updates
        # until accuracy is great enough so at testing it would be at least 81%
        accuracy = 0
        while accuracy < 0.88: # funciona ate com 0.85, deixando 88 pra ficar safe
            
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                
                parameters = [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3]
                grad_wrt_W1, \
                grad_wrt_W2, \
                grad_wrt_W3, \
                grad_wrt_b1, \
                grad_wrt_b2, \
                grad_wrt_b3 = nn.gradients(loss, parameters)
                
                self.W1.update(grad_wrt_W1, -self.learning_rate)
                self.W2.update(grad_wrt_W2, -self.learning_rate)
                self.W3.update(grad_wrt_W3, -self.learning_rate)
                self.b1.update(grad_wrt_b1, -self.learning_rate)
                self.b2.update(grad_wrt_b2, -self.learning_rate)
                self.b3.update(grad_wrt_b3, -self.learning_rate)
            
            accuracy = dataset.get_validation_accuracy()
            # print(accuracy)

