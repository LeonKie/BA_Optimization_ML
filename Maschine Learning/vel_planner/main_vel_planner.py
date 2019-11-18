from sym_qp import SymQP  # TODO: must NOT depend on chosen objective function
import numpy as np
import json
import JsonExample 

sqp_stgs = {'m': 100,  # number of discretization points
             'b_online_mode': 0,  # 0 or 1 for running on car or not
             'obj_func': 'objdv',  # 'slacks' or 'objdv'
             'ev_vel_w': 10  # weight on every slack variable
             }

ADDTOOTRAINING=False

# SQP termination criterion
err_ = 0.01 * sqp_stgs['m']



vini_mps_ = 60
x0_ = [vini_mps_]  # Initial velocity
x0_.extend([60] * (sqp_stgs['m'] - 1))  # Velocity guess
if sqp_stgs['obj_func'] == 'slacks':
    x0_.extend([0] * sqp_stgs['m'])  # slack variables on velocity
kappa_ = [0.0] * sqp_stgs['m']
delta_s_ = [2] * (sqp_stgs['m'] - 1)




# Create SymQP-instance
if sqp_stgs['obj_func'] == 'slacks':
    symqp = SymQP(sqp_stgs['m'],
                  sqp_stgs['obj_func'],
                  ev_vel_w=sqp_stgs['ev_vel_w'])
elif sqp_stgs['obj_func'] == 'objdv':
    symqp = SymQP(sqp_stgs['m'],
                  sqp_stgs['obj_func'])

# --- Call substitute_symbolics to retrieve lambda-functions from symbolic expressions
symqp.subs_symbolics()

symqp.osqp_init(x0_, vini_mps_, kappa_, delta_s_, sqp_stgs['b_online_mode'])

# --- Sequential calls of QP
err = np.inf
while err > err_:
    # --- Update parameters of QP
    symqp.osqp_update_online(x0_, vini_mps_, kappa_, delta_s_)
    sol = symqp.osqp_solve()

    # --- Solution = QP-solution + initial operating-point
    v_op_ = sol + np.array(x0_)
    print('Optimized velocity profile: ', v_op_[0:sqp_stgs['m']])
    if sqp_stgs['obj_func'] == 'slacks':
        print('Slacks on velocity: ', v_op_[sqp_stgs['m']:2 * sqp_stgs['m']])

    # --- Create new operating-point
    x0_old_ = x0_
    x0_fst_ = x0_[0]
    x0_tmp_ = sol[1:np.size(x0_)] + np.array(x0_)[1:np.size(x0_)]
    x0_ = np.append(x0_fst_, x0_tmp_)

    # --- Calculate SQP iteration error
    # err = 0
    err = np.matmul(v_op_ - x0_old_, v_op_ - x0_old_)


    x0_ = x0_.tolist()

# --- Export Data 
if ADDTOOTRAINING:
    data_out={"V_op":list(v_op_), "Kappa": list(kappa_), "delta_s": list(delta_s_),"v_ini": vini_mps_}

    TrainingExampleNR=JsonExample.saveData(data_out)
    print(TrainingExampleNR, " Exporting Finished")

# --- Visualize SQP-solution
if not sqp_stgs['b_online_mode']:
    symqp.vis_sol(v_op_, kappa_, delta_s_)

