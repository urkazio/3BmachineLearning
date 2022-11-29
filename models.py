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
        Deberiais obtener el producto escalar (o producto punto) que es "equivalente" a la distancia del coseno
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(x, self.w)



    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.
        Dependiendo del valor del coseno devolvera 1 o -1

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1



    def train(self, dataset):
        """
        Train the perceptron until convergence.
        Hasta que TODOS los ejemplos del train esten bien clasificados. Es decir, hasta que la clase predicha en
        se corresponda con la real en TODOS los ejemplos del train
        """
        "*** YOUR CODE HERE ***"
        convergencia = False
        batch_size = 1
        while not convergencia:
            convergencia = True  # momentaneo
            for x, y in dataset.iterate_once(batch_size):
                yp = self.get_prediction(x)
                if yp != nn.as_scalar(y):
                    convergencia = False
                    self.w.update(x, nn.as_scalar(y))  # coño no entendia q funcionaba asi




class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    NO ES CLASIFICACION, ES REGRESION. ES DECIR; APRENDER UNA FUNCION.
    SI ME DAN X TENGO QUE APRENDER A OBTENER LA MISMA Y QUE EN LA FUNCION ORIGINAL DE LA QUE QUIERO APRENDER
    """
    def __init__(self):
        # Initialize your model parameters here
        self.batch_size = 20
        self.lr = -0.002

        # entrada
        self.w0 = nn.Parameter(1, 15)
        self.b0 = nn.Parameter(1, 15)

        # intermedia 1
        self.w1 = nn.Parameter(15, 10)
        self.b1 = nn.Parameter(1, 10)

        # intermedia 2
        self.w2 = nn.Parameter(10, 5)
        self.b2 = nn.Parameter(1, 5)

        # salida
        self.w3 = nn.Parameter(5, 1)
        self.b3 = nn.Parameter(1, 1)



    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1). En este caso cada ejemplo solo esta compuesto por un rasgo
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values.
            Como es un modelo de regresion, cada valor y tambien tendra un unico valor
        """
        "*** YOUR CODE HERE ***"

        # nn.Linear() realiza la multiplicacion matricial
        # ejecucion feedForward !!! y devuelve el array de probabilidades de cada clase

        capa1 = nn.AddBias(nn.Linear(x, self.w0), self.b0)
        capa2 = nn.AddBias(nn.Linear(nn.ReLU(capa1), self.w1), self.b1)
        capa3 = nn.AddBias(nn.Linear(nn.ReLU(capa2), self.w2), self.b2)
        capa4 = nn.AddBias(nn.Linear(nn.ReLU(capa3), self.w3), self.b3)
        return capa4



    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
                ----> ES FACIL COPIA Y PEGA ESTO Y ANNADE LA VARIABLE QUE HACE FALTA PARA CALCULAR EL ERROR
                return nn.SquareLoss(self.run(x),ANNADE LA VARIABLE QUE ES NECESARIA AQUI), para medir el error, necesitas comparar el resultado de tu prediccion con .... que?
        """

        # error cuadratico tras una iteracion

        return nn.SquareLoss(self.run(x), y)



    def train(self, dataset):
        """
        Trains the model.

        """
        batch_size = self.batch_size
        total_loss = 100000
        while total_loss > 0.01: # early stop ---> objetivo bajar de 0.02 para q2:6/6

            #ITERAR SOBRE EL TRAIN EN LOTES MARCADOS POR EL BATCH SIZE COMO HABEIS HECHO EN LOS OTROS EJERCICIOS
            #ACTUALIZAR LOS PESOS EN BASE AL ERROR loss = self.get_loss(x, y) QUE RECORDAD QUE GENERA
            #UNA FUNCION DE LA LA CUAL SE  PUEDE CALCULAR LA DERIVADA (GRADIENTE)

            for x, y in dataset.iterate_once(batch_size):
                total_loss = self.get_loss(x, y)
                """
                --> gradients(loss, parameters):

                Usage: nn.gradients(loss, parameters)
                Inputs:
                    loss: a SquareLoss or SoftmaxLoss node
                    parameters: a list (or iterable) containing Parameter nodes
                Output: a list of Constant objects, representing the gradient of the loss
                    with respect to each provided parameter.
                """

                # obtiene una lista de constantes por cada parametro de entrada en la lista
                # dicha constante representa el gradiente de la perdida respecto a cada parametro
                gradientes = nn.gradients(total_loss, [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])

                # teniendo el gradiente de cada param. actualizamos todas las variables
                self.w0.update(gradientes[0], self.lr)
                self.b0.update(gradientes[1], self.lr)
                self.w1.update(gradientes[2], self.lr)
                self.b1.update(gradientes[3], self.lr)
                self.w2.update(gradientes[4], self.lr)
                self.b2.update(gradientes[5], self.lr)
                self.w3.update(gradientes[6], self.lr)
                self.b3.update(gradientes[7], self.lr)

                # convertirlo a escalar para luego poder hacer la comparacion con int
                total_loss = nn.as_scalar(total_loss)





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
        # TEN EN CUENTA QUE TIENES 10 CLASES, ASI QUE LA ULTIMA CAPA TENDRA UNA SALIDA DE 10 VALORES,
        # UN VALOR POR CADA CLASE

        output_size = 10  # TAMAÑO EQUIVALENTE AL NUMERO DE CLASES DADO QUE QUIERES OBTENER 10 CLASES
        pixel_dim_size = 28
        pixel_vector_length = pixel_dim_size* pixel_dim_size

        self.batch_size = 20
        self.lr = -0.002

        # entrada
        self.w0 = nn.Parameter(pixel_vector_length, 100)
        self.b0 = nn.Parameter(1, 100)

        # intermedia 1
        self.w1 = nn.Parameter(100, 100)
        self.b1 = nn.Parameter(1, 100)

        # intermedia 2
        self.w2 = nn.Parameter(100, 100)
        self.b2 = nn.Parameter(1, 100)

        # salida
        self.w3 = nn.Parameter(100, output_size)
        self.b3 = nn.Parameter(1, output_size)



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
            output_size = 10 # TAMANO EQUIVALENTE AL NUMERO DE CLASES DADO QUE QUIERES OBTENER 10 "COSENOS"
        """
        capa1 = nn.AddBias(nn.Linear(x, self.w0), self.b0)
        capa2 = nn.AddBias(nn.Linear(nn.ReLU(capa1), self.w1), self.b1)
        capa3 = nn.AddBias(nn.Linear(nn.ReLU(capa2), self.w2), self.b2)
        capa4 = nn.AddBias(nn.Linear(nn.ReLU(capa3), self.w3), self.b3)
        return capa4



    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).
        POR EJEMPLO: [0,0,0,0,0,1,0,0,0,0,0] seria la y correspondiente al 5
                     [0,1,0,0,0,0,0,0,0,0,0] seria la y correspondiente al 1

        EN ESTE CASO ESTAMOS HABLANDO DE MULTICLASS, ASI QUE TIENES QUE CALCULAR
        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        # NO ES NECESARIO QUE LO IMPLEMENTEIS, SE OS DA HECHO
        return nn.SoftmaxLoss(self.run(x), y)

        # COMO VEIS LLAMA AL RUN PARA OBTENER POR CADA BATCH
        # LOS 10 VALORES DEL "COSENO". TENIENDO EL Y REAL POR CADA EJEMPLO
        # APLICA SOFTMAX PARA CALCULAR LA PROBABILIDA MAX
        # Y ESA SERA SU PREDICCION,
        # LA CLASE QUE MUESTRE EL MAYOR PROBABILIDAD, LA PREDICCION MAS PROBABLE, Y LUEGO LA COMPARARA CON Y


    def train(self, dataset):
        """
        Trains the model.
        EN ESTE CASO EN VEZ DE PARAR CUANDO EL ERROR SEA MENOR QUE UN VALOR O NO HAYA ERROR (CONVERGENCIA),
        SE PUEDE HACER ALGO SIMILAR QUE ES EN NUMERO DE ACIERTOS. EL VALIDATION ACCURACY
        NO LO TENEIS QUE IMPLEMENTAR, PERO SABED QUE EMPLEA EL RESULTADO DEL SOFTMAX PARA CALCULAR
        EL NUM DE EJEMPLOS DEL TRAIN QUE SE HAN CLASIFICADO CORRECTAMENTE
        """
        batch_size = self.batch_size
        while dataset.get_validation_accuracy() < 0.97:
            #ITERAR SOBRE EL TRAIN EN LOTES MARCADOS POR EL BATCH SIZE COMO HABEIS HECHO EN LOS OTROS EJERCICIOS
            #ACTUALIZAR LOS PESOS EN BASE AL ERROR loss = self.get_loss(x, y) QUE RECORDAD QUE GENERA
            #UNA FUNCION DE LA LA CUAL SE  PUEDE CALCULAR LA DERIVADA (GRADIENTE)
            for x, y in dataset.iterate_once(self.batch_size):
                total_loss = self.get_loss(x, y)
                gradientes = nn.gradients(total_loss, [self.w0, self.b0, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])

                # teniendo el gradiente de cada param. actualizamos todas las variables
                self.w0.update(gradientes[0], self.lr)
                self.b0.update(gradientes[1], self.lr)
                self.w1.update(gradientes[2], self.lr)
                self.b1.update(gradientes[3], self.lr)
                self.w2.update(gradientes[4], self.lr)
                self.b2.update(gradientes[5], self.lr)
                self.w3.update(gradientes[6], self.lr)
                self.b3.update(gradientes[7], self.lr)




