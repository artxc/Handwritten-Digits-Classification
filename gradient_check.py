import numpy as np


def check_gradient(f, x, delta=1e-5, tol=1e-4):
    _, analytic_grad = f(x)

    delta_vector = np.zeros(x.shape)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        analytic_grad_at_ix = analytic_grad[ix]

        delta_vector[ix] = delta
        numeric_grad_at_ix = (f(x + delta_vector)[0] - f(x - delta_vector)[0]) / (2 * delta)
        delta_vector[ix] = 0.0

        if not np.isclose(numeric_grad_at_ix, analytic_grad_at_ix, tol):
            print(f'Gradients are different at {ix}. Analytic: {analytic_grad_at_ix}, Numeric: {numeric_grad_at_ix}')
            return False

        it.iternext()

    print('Gradient check passed!')
    return True


def check_model_gradient(model, X, y, loss_and_gradient, delta=1e-5, tol=1e-4):
    for parameter in model.parameters():
        print(f'Checking gradient. W/B shape: {parameter.value.shape}')

        initial_w = parameter.value

        def helper_func(w):
            parameter.value = w
            parameter.grad = np.zeros_like(parameter.value)
            loss, grad = loss_and_gradient(model(X), y)
            model.backward(grad)
            return loss, parameter.grad

        check_gradient(helper_func, initial_w, delta, tol)


def check_layer_parameter_gradient(layer, x, parameter_id, delta=1e-5, tol=1e-4):
    parameter = layer.parameters()[parameter_id]
    initial_w = parameter.value

    output_weight = np.random.randn(*layer.forward(x).shape)

    def helper_func(w):
        parameter.value = w
        parameter.grad = np.zeros_like(parameter.value)
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        d_out = output_weight
        layer.backward(d_out)
        return loss, parameter.grad

    return check_gradient(helper_func, initial_w, delta, tol)


def check_layer_gradient(layer, x, delta=1e-5, tol=1e-4):
    output_weight = np.random.randn(*layer.forward(x).shape)

    def helper_func(x):
        output = layer.forward(x)
        loss = np.sum(output * output_weight)
        d_out = output_weight
        grad = layer.backward(d_out)
        return loss, grad

    return check_gradient(helper_func, x, delta, tol)
