#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    def parameter_shift_term(circuit, weights, i, j):

        if j !=0:
            shifted = weights.copy()
            shifted[i] += np.pi/2
            forward = circuit(shifted)
            shifted[i] -= np.pi
            backward = circuit(shifted) # backward evaluation

            shifted[j] += np.pi/2
            backward_forward = circuit(shifted)  # forward evaluation
            shifted[j] -= np.pi
            backward_backward = circuit(shifted)
            shifted[i] +=np.pi
            forward_backward = circuit(shifted)
            shifted[j] += np.pi
            forward_forward =  circuit(shifted)
        
            return 0.5 * (forward - backward), 0.25*(forward_forward+backward_backward-forward_backward-backward_forward)
        elif i == j:
            shifted = weights.copy()
            regular = circuit(shifted)
            shifted[i] += np.pi/2
            forward = circuit(shifted)
            shifted[i] += np.pi/2
            forward_forward = circuit(shifted)
            shifted[i] -= 3*np.pi/2
            backward = circuit(shifted) # backward evaluation
            shifted[i] -= np.pi/2
            backward_backward  = circuit(shifted)

            return 0.5 * (forward - backward), 0.25*(forward_forward+backward_backward-2*regular)
        
        else:
            shifted = weights.copy()
            shifted[i] += np.pi/2
            forward = circuit(shifted)
            shifted[i] -= np.pi
            backward = circuit(shifted) # backward evaluation
            return 0.5*(forward - backward), 0

    for i in range(len(weights)):
        gradient[i] = parameter_shift_term(circuit, weights, i, 0)[0]
        if gradient[i] == 0:
           hessian[:][i] = hessian[i][:]=0
        else:
            for j in range(5-i):
                hessian[i][j] = parameter_shift_term(circuit, weights, i, j)[1] 
                hessian[j][i] = hessian[i][j]

    # QHACK #
    return gradient, hessian, circuit.diff_options["method"]

if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
