#Multiple Class Logistic Regression

 Some Theory - It's also known as Multi-class Classification problem. This algorithm tries to predict discrete value of
               variable y which is used to classify the data

##Algorithm -

 Assumption - Let, function J(Q0,Q1,....,Qn)
              we will use sigmoid function for computing the hypothesis
              cost function = -1/m * (Y*log(hQ(X)) + (1-Y)*log(1 - hQ(X)))
              This cost function gives us convex graph if plotted between iterations & Q
              here, caps X & Y refer to matrix for eg -> X = [x0, x1, .., xm]

 Aim - To find minJ(Q0,Q1,....,Qn)

 - Begin with any value Q0, Q1, ...., Qn
 - Keep changing values of Q0,Q1,....,Qn to reduce J(Q0,Q1,....,Qn), until we find min J(Q0,Q1,....,Qn)

 Repeat till local minima is achieved {
        Q = Q - learning_rate * sum from i=0 to i=m( h(Q)(i) - Y(i) ) * X(i)
        calculate cost with each iteration to check if Gradient descent is converging properly
    }

##How It Works

Add your training data in list TRAINING_DATA present in file training_data.py

Add your new data in list PREDICTION_DATA present in file training_data.py

Run Command - python logistic_regression.py

you can change LEARNING_RATE present in logistic_regression.py

##TROUBLESHOOTING -

If you are getting new values of Q0, Q1,...., Qn positive and negative alternatively. Try, reducing Learning rate present in file logistic_regression.py