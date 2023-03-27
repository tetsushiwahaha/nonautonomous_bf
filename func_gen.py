import numpy as np
import sympy as sp
import os

# fmt: off
# DO NOT EDIT ABOVE

# system description
xdim = 3
pdim = 4

def func(t, x, p):
    f = [
         x[1],
         -(x[0]**2 + 3*x[2]**2)*x[0]/8 + sp.cos(t)*p[3] - p[0]*x[1],
         -(3*x[0]**2 + x[2]**2)*p[1]*x[2]/8 + p[2]]
    return sp.Matrix(f)

# fmt: on
# DO NOT EDIT BELOW

header = """\
#include "dynamical_system.hpp"

void dynamical_system::sys_func(const Eigen::VectorXd &x, const double /*t*/) {
"""

sym_t = sp.Symbol("t")
sym_x = sp.MatrixSymbol("x", xdim, 1)
sym_p = sp.MatrixSymbol("p", pdim, 1)

n = xdim

code = ""

code += "import numpy as np\n\n"
code += "def sys_func(t, x, p, data):\n"

f = func(sym_t, sym_x, sym_p)
code += "\tf = "
code += str(list(f))
code += "\n"

dfdx = sp.derive_by_array(
    [f[i] for i in range(xdim)], [sym_x[i] for i in range(xdim)]
).transpose()
code += "\tdfdx = np.array("
code += str(dfdx)
code += ")\n"


for idx_param in range(pdim):
    if idx_param == 0:
        code += '\tif data.dic["variable_param"] == 0:\n'
    else:
        code += '\telif data.dic["variable_param"] == ' + str(idx_param) + ":\n"
    code += "\t\t"
    dfdlambda = sp.diff(f, sym_p[idx_param])
    code += "dfdl = np.array("
    code += str(list(dfdlambda))
    code += ")\n"


dfdxdx = [sp.zeros(xdim, xdim) for j in range(xdim)]
for i in range(xdim):
    dfdxdx[i] = sp.diff(dfdx, sym_x[i])
code += "\td2fdx2 = np.array("
code += str(dfdxdx)
code += ")\n"

for idx_param in range(pdim):
    if idx_param == 0:
        code += '\tif data.dic["variable_param"] == 0:\n'
    else:
        code += '\telif data.dic["variable_param"] == ' + str(idx_param) + ":\n"
    code += "\t\t"
    dfdxdlambda = sp.diff(dfdx, sym_p[idx_param])
    code += "d2fdxdl = np.array("
    code += str(dfdxdlambda)
    code += ")\n"

code += "\treturn f, dfdx, dfdl, d2fdx2, d2fdxdl\n"

for i in range(xdim):
    code = code.replace("x[" + str(i) + ", 0]", "x[" + str(i) + "]")
for i in range(pdim):
    code = code.replace("p[" + str(i) + ", 0]", "p[" + str(i) + "]")
code = code.replace("cos(t)", "np.cos(t)")
code = code.replace("log(", "np.log(")

f = open("sys_func.py", "w")
f.write(code)
f.close()

os.system("black sys_func.py")
