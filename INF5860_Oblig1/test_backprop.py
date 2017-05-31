import numpy as np
import nose

from code.classifiers.fc_net import  import TwoLayerNet
from code.data_utils import  import get_CIFAR10_data
from code.layer_utils import affine_relu_forward, affine_relu_backward
from code.layers import *
from code.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from code.solver import Solver

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def test_affine_forward():
  num_inputs = 2
  input_shape = (4, 5, 6)
  output_dim = 3

  input_size = num_inputs * np.prod(input_shape)
  weight_size = output_dim * np.prod(input_shape)

  x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
  w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
  b = np.linspace(-0.3, 0.1, num=output_dim)
  out, _ = affine_forward(x, w, b)
  correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                          [3.25553199, 3.5141327, 3.77273342]])
  assert rel_error(out, correct_out) < 1e-9


def test_affine_backward():
  dx_sum, dw_sum, db_sum = 0, 0, 0
  N = 20
  np.random.seed(7)
  for i in range(N):
    dx_err, dw_err, db_err = run_singel_test()
    assert dx_err < 1e-7
    assert dw_err < 1e-7
    assert db_err < 1e-7
    dx_sum += dx_err; dw_sum += dw_err; db_sum += db_err
  assert (dx_sum/float(N)) < 1e-9
  assert (dw_sum/float(N)) < 1e-9
  assert (db_sum/float(N)) < 1e-9


def test_relu_forward():
  x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

  out, _ = relu_forward(x)
  correct_out = np.array([[0., 0., 0., 0., ],
                          [0., 0., 0.04545455, 0.13636364, ],
                          [0.22727273, 0.31818182, 0.40909091, 0.5, ]])
  assert rel_error(out, correct_out) < 1e-7


def test_relu_backward():
  x = np.random.randn(10, 10)
  dout = np.random.randn(*x.shape)

  dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

  _, cache = relu_forward(x)
  dx = relu_backward(dout, cache)

  # The error should be around 1e-12
  assert rel_error(dx_num, dx) < 1e-11


def test_relu_affine_forward_backward():
  x = np.random.randn(2, 3, 4)
  w = np.random.randn(12, 10)
  b = np.random.randn(10)
  dout = np.random.randn(2, 10)

  out, cache = affine_relu_forward(x, w, b)
  dx, dw, db = affine_relu_backward(dout, cache)

  dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
  dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
  db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)

  assert rel_error(dx_num, dx) < 1e-9
  assert rel_error(dx_num, dx) < 1e-9
  assert rel_error(dx_num, dx) < 1e-9


def test_svm_loss():
  num_classes, num_inputs = 10, 50
  x = 0.001 * np.random.randn(num_inputs, num_classes)
  y = np.random.randint(num_classes, size=num_inputs)

  dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
  loss, dx = svm_loss(x, y)
  assert rel_error(dx_num, dx) < 1e-9


def test_softmax_loss():
  num_classes, num_inputs = 10, 50
  x = 0.001 * np.random.randn(num_inputs, num_classes)
  y = np.random.randint(num_classes, size=num_inputs)

  dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
  loss, dx = softmax_loss(x, y)
  assert rel_error(dx_num, dx) < 1e-8


def test_two_layer_net_initialization():
  N, D, H, C = 3, 5, 50, 7
  X = np.random.randn(N, D)
  y = np.random.randint(C, size=N)

  std = 1e-2
  model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)
  W1_std = abs(model.params['W1'].std() - std)
  b1 = model.params['b1']
  W2_std = abs(model.params['W2'].std() - std)
  b2 = model.params['b2']
  assert W1_std < std / 10, 'First layer weights do not seem right'
  assert np.all(b1 == 0), 'First layer biases do not seem right'
  assert W2_std < std / 10, 'Second layer weights do not seem right'
  assert np.all(b2 == 0), 'Second layer biases do not seem right'


def test_two_layer_net_forwardpass():
  N, D, H, C = 3, 5, 50, 7
  X = np.random.randn(N, D)
  y = np.random.randint(C, size=N)

  std = 1e-2
  model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)
  W1_std = abs(model.params['W1'].std() - std)
  b1 = model.params['b1']
  W2_std = abs(model.params['W2'].std() - std)
  b2 = model.params['b2']

  model.params['W1'] = np.linspace(-0.7, 0.3, num=D * H).reshape(D, H)
  model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
  model.params['W2'] = np.linspace(-0.3, 0.4, num=H * C).reshape(H, C)
  model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
  X = np.linspace(-5.5, 4.5, num=N * D).reshape(D, N).T
  scores = model.loss(X)
  correct_scores = np.asarray(
    [[11.53165108, 12.2917344, 13.05181771, 13.81190102, 14.57198434, 15.33206765, 16.09215096],
     [12.05769098, 12.74614105, 13.43459113, 14.1230412, 14.81149128, 15.49994135, 16.18839143],
     [12.58373087, 13.20054771, 13.81736455, 14.43418138, 15.05099822, 15.66781506, 16.2846319]])
  scores_diff = np.abs(scores - correct_scores).sum()
  assert scores_diff < 1e-6, 'Problem with test-time forward pass'


def test_two_layer_net_training_loss():
  N, D, H, C = 3, 5, 50, 7
  X = np.random.randn(N, D)
  y = np.random.randint(C, size=N)

  std = 1e-2
  model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)
  W1_std = abs(model.params['W1'].std() - std)
  b1 = model.params['b1']
  W2_std = abs(model.params['W2'].std() - std)
  b2 = model.params['b2']

  model.params['W1'] = np.linspace(-0.7, 0.3, num=D * H).reshape(D, H)
  model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
  model.params['W2'] = np.linspace(-0.3, 0.4, num=H * C).reshape(H, C)
  model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
  X = np.linspace(-5.5, 4.5, num=N * D).reshape(D, N).T
  scores = model.loss(X)
  correct_scores = np.asarray(
    [[11.53165108, 12.2917344, 13.05181771, 13.81190102, 14.57198434, 15.33206765, 16.09215096],
     [12.05769098, 12.74614105, 13.43459113, 14.1230412, 14.81149128, 15.49994135, 16.18839143],
     [12.58373087, 13.20054771, 13.81736455, 14.43418138, 15.05099822, 15.66781506, 16.2846319]])
  scores_diff = np.abs(scores - correct_scores).sum()

  y = np.asarray([0, 5, 1])
  loss, grads = model.loss(X, y)
  correct_loss = 3.4702243556
  assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'


def test_two_layer_net_training_loss_regularization():
  N, D, H, C = 3, 5, 50, 7
  X = np.random.randn(N, D)
  y = np.random.randint(C, size=N)

  std = 1e-2
  model = TwoLayerNet(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)
  W1_std = abs(model.params['W1'].std() - std)
  b1 = model.params['b1']
  W2_std = abs(model.params['W2'].std() - std)
  b2 = model.params['b2']

  model.params['W1'] = np.linspace(-0.7, 0.3, num=D * H).reshape(D, H)
  model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
  model.params['W2'] = np.linspace(-0.3, 0.4, num=H * C).reshape(H, C)
  model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
  X = np.linspace(-5.5, 4.5, num=N * D).reshape(D, N).T
  scores = model.loss(X)
  correct_scores = np.asarray(
    [[11.53165108, 12.2917344, 13.05181771, 13.81190102, 14.57198434, 15.33206765, 16.09215096],
     [12.05769098, 12.74614105, 13.43459113, 14.1230412, 14.81149128, 15.49994135, 16.18839143],
     [12.58373087, 13.20054771, 13.81736455, 14.43418138, 15.05099822, 15.66781506, 16.2846319]])
  scores_diff = np.abs(scores - correct_scores).sum()

  y = np.asarray([0, 5, 1])
  model.reg = 1.0
  loss, grads = model.loss(X, y)
  correct_loss = 26.5948426952
  assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'


def test_two_layer_net_solver():
  model = TwoLayerNet()
  solver = None

  data = get_CIFAR10_data()
  solver = Solver(model, data, update_rule='sgd', optim_config={'learning_rate': 1e-3}, lr_decay=0.95)
  solver.train()

def run_singel_test():
  x = np.random.randn(10, 2, 3)
  w = np.random.randn(6, 5)
  b = np.random.randn(5)
  dout = np.random.randn(10, 5)
  dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
  dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
  db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)
  _, cache = affine_forward(x, w, b)
  dx, dw, db = affine_backward(dout, cache)
  # The error should be around 1e-10
  return rel_error(dx_num, dx), rel_error(dw_num, dw), rel_error(db_num, db)