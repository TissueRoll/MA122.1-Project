Input: error rates from output layer
Output: new set of weights
BackPropagation(error_rates):
	# first step is to compute for weights between output layer and last hidden layer
	# yi represents the ith value in the output layer
	# f'(x) represents the derivative of the activation function for the current layer
	derivatives = f'(y[i]) for i in each computed output value
	gradients = error_rates[i] * derivatives[i] for i in each neuron #result is n x 1 matrix
	delta_w = gradients * last_hidden_layer #last_hidden_layer transposed to become row matrix
	new_weights_to_output = previous_weights_to_output - delta_w #previous weights between last hidden layer and output layer is replaced with new_weights

	# succeeding steps for each hidden layer until the input
	gradients = transpose(gradients)
	current_layer = last_hidden_layer
	while current_layer != input_layer:
		derivatives = f'(current_layer[i]) for i in each computed value in current_layer
		gradients = gradients * transpose(previous_weights)
		gradients[i] = gradients[i] * derivatives[i]
		delta_w = left_layer * gradients #left_layer is the layer to the left of current_layer; expressed as a column vector
		new_weights_to_current = previous_weights_to_current - delta_w
		current_layer = left_layer
	
	return set of new_weights
