import numpy as np
import pytest

from aegis.qgate.kernel import gram, kernel_value, quantum_state
from aegis.qgate.kernel import k_hardware_style


def test_state_vector_normalized():
    x = np.array([
        0.2, 0.7, 1.3, 2.0,
        0.5, 1.1, 2.2, 0.8
    ])

    state = quantum_state(x)

    assert state.shape == (256,)
    assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-9)


def test_kernel_self_similarity_is_one():
    x = np.array([
        0.2, 0.7, 1.3, 2.0,
        0.5, 1.1, 2.2, 0.8
    ])

    assert np.isclose(kernel_value(x, x), 1.0, atol=1e-9)


def test_kernel_is_symmetric():
    x = np.array([
        0.2, 0.7, 1.3, 2.0,
        0.5, 1.1, 2.2, 0.8
    ])

    y = np.array([
        1.0, 0.5, 2.1, 0.9,
        0.6, 1.3, 0.7, 1.4
    ])

    assert np.isclose(
        kernel_value(x, y),
        kernel_value(y, x),
        atol=1e-9
    )


def test_gram_matrix_diagonal_is_one():
    X = np.array([
        [
            0.2, 0.7, 1.3, 2.0,
            0.5, 1.1, 2.2, 0.8
        ],
        [
            1.0, 0.5, 2.1, 0.9,
            0.6, 1.3, 0.7, 1.4
        ],
        [
            2.0, 1.4, 0.8, 0.3,
            1.2, 0.9, 1.6, 0.4
        ],
    ])

    K = gram(X)

    assert K.shape == (3, 3)
    assert np.allclose(np.diag(K), 1.0, atol=1e-9)


def test_invalid_input_rejected():

    # Wrong number of features
    with pytest.raises(ValueError):
        quantum_state([
            0.1, 0.2, 0.3
        ])

    # Value outside [0, pi]
    with pytest.raises(ValueError):
        quantum_state([
            0.1, 0.2, 0.3, 4.0,
            0.5, 1.0, 1.5, 2.0
        ])


def test_kernel_values_are_bounded():

    x = np.array([
        0.2, 0.7, 1.3, 2.0,
        0.5, 1.1, 2.2, 0.8
    ])

    y = np.array([
        1.0, 0.5, 2.1, 0.9,
        0.6, 1.3, 0.7, 1.4
    ])

    value = kernel_value(x, y)

    assert 0.0 <= value <= 1.0


def test_hardware_style_matches_state_vector_kernel():

    x = np.array([
        0.2, 0.8, 1.4, 2.1,
        0.5, 1.1, 2.2, 0.8
    ])

    y = np.array([
        2.5, 1.7, 0.4, 1.2,
        1.3, 0.9, 1.6, 0.4
    ])

    expected = kernel_value(x, y)
    actual = k_hardware_style(x, y)

    assert np.isclose(actual, expected, atol=1e-9)