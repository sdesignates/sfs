import torch
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
import scipy.linalg

class MatrixSquareRoot(Function):
    """Square root of a positive definite matrix.
       Given a positive semi-definite matrix X,
       X = X^{1/2}X^{1/2}, compute the gradient: dX^{1/2} by solving the Sylvester equation, 
       dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
    """
    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real)#.type_as(input)
        ctx.save_for_backward(sqrtm) # save in cpu
        sqrtm = sqrtm.type_as(input)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)
            grad_input = torch.from_numpy(grad_sqrtm).type_as(grad_output.data)
        return Variable(grad_input)

