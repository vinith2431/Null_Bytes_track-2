
import numpy as np
import pennylane as qml


N_QUBITS = 4
WIRES = list(range(N_QUBITS))

dev = qml.device("default.qubit", wires=N_QUBITS)


def feature_map(x):
    """
    Encode four input features into a four-qubit quantum state.

    Input:
        x: Four real-valued features, each in [0, pi].

    Feature map:
        1. Apply Hadamard gates.
        2. Encode each feature using RZ(x_i).
        3. Apply nearest-neighbor entangling blocks:
           CNOT -> RZ((pi - x_i)*(pi - x_j)) -> CNOT.

    The input is expected to be preprocessed by embed.py.
    """
    x = np.asarray(x, dtype=float)

    if x.shape != (N_QUBITS,):
        raise ValueError("Input must contain exactly four features.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Input features must be finite.")

    if np.any(x < 0) or np.any(x > np.pi):
        raise ValueError("Input features must be in [0, pi].")

    for wire in WIRES:
        qml.Hadamard(wires=wire)

    for wire in WIRES:
        qml.RZ(x[wire], wires=wire)

    for i in range(N_QUBITS - 1):
        j = i + 1
        angle = (np.pi - x[i]) * (np.pi - x[j])

        qml.CNOT(wires=[i, j])
        qml.RZ(angle, wires=j)
        qml.CNOT(wires=[i, j])


@qml.qnode(dev)
def _state_circuit(x):
    """Return the state prepared by the feature map."""
    feature_map(x)
    return qml.state()


def quantum_state(x):
    """Return the four-qubit state vector for input x."""
    return np.asarray(_state_circuit(x), dtype=complex)


def kernel_value(x, y):
    """
    Calculate the fidelity kernel:
        K(x, y) = |<phi(x)|phi(y)>|^2
    """
    state_x = quantum_state(x)
    state_y = quantum_state(y)

    overlap = np.vdot(state_x, state_y)
    value = float(np.abs(overlap) ** 2)

    # Protect against negligible floating-point excursions.
    return float(np.clip(value, 0.0, 1.0))


def gram(X, Y=None):
    """
    Construct a kernel matrix.

    If Y is None, compute the square Gram matrix K(X, X).
    Otherwise, compute the cross-kernel matrix K(X, Y).
    """
    X = np.asarray(X, dtype=float)

    if X.ndim != 2 or X.shape[1] != N_QUBITS:
        raise ValueError("X must have shape (n_samples, 4).")

    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y, dtype=float)

        if Y.ndim != 2 or Y.shape[1] != N_QUBITS:
            raise ValueError("Y must have shape (n_samples, 4).")

    states_X = [quantum_state(row) for row in X]
    states_Y = states_X if Y is X else [quantum_state(row) for row in Y]

    matrix = np.empty((len(X), len(Y)), dtype=float)

    for i, state_x in enumerate(states_X):
        for j, state_y in enumerate(states_Y):
            matrix[i, j] = np.abs(np.vdot(state_x, state_y)) ** 2

    return np.clip(matrix, 0.0, 1.0)


@qml.qnode(dev)
def _overlap_circuit(x, y):
    """Compute the overlap using U(y)† U(x) and return all-zero probability."""
    feature_map(x)
    qml.adjoint(feature_map)(y)
    return qml.probs(wires=WIRES)


def k_hardware_style(x, y):
    """
    Independently calculate the fidelity kernel using an overlap circuit.

    For normalized pure states:
        K(x, y) = |<phi(y)|phi(x)>|^2

    Applying U(y)† after U(x) makes the all-zero probability
    equal to the squared state overlap.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Reuse the input validation in feature_map.
    if x.shape != (N_QUBITS,) or y.shape != (N_QUBITS,):
        raise ValueError("Both inputs must contain exactly four features.")

    # Validate each vector before executing the circuit.
    for vector in (x, y):
        if not np.all(np.isfinite(vector)):
            raise ValueError("Input features must be finite.")
        if np.any(vector < 0) or np.any(vector > np.pi):
            raise ValueError("Input features must be in [0, pi].")

    probabilities = np.asarray(_overlap_circuit(x, y), dtype=float)
    return float(np.clip(probabilities[0], 0.0, 1.0))
