import sympy as sym
import numpy as np
import osqp as osqp  # QP solver from University of Oxford for embedded problems with high performance
import scipy.sparse as sparse
from matplotlib import pyplot as plt

########################################################################################################################
# Activate virtual env. 'conda_qp' to run this code
########################################################################################################################
# TODO: introduce only a few slack variables with short ... longer ... long horizon
# TODO: simplify pos/neg-constraints to box-constraints
# TODO: make vmax an online-parameter (for follow mode)
# TODO: make axmax and aymax online-parameters
# TODO: integrate limits in constraint-plots
# TODO: check velocity box constraints
# TODO: Adapt slacks-objective-code to d_s_m-vector


class SymQP():

    def __init__(self, m, obj_func, ev_vel_w=None):
        self.m = m
        self.obj_func = obj_func
        self.ev_vel_w = ev_vel_w
        print('*** --- Chosen optimizer-option:', obj_func, '--- ***')

        self.init_symbolics()
        self.sol_osqp = osqp.OSQP()


    def init_symbolics(self):
        ################################################################################################################
        # Dimensions
        ################################################################################################################
        m = self.m
        ev_vel_w = self.ev_vel_w

        ################################################################################################################
        # Matrices
        ################################################################################################################
        D_mat = sym.Matrix(- np.eye(m) + np.eye(m, k=1))
        D_mat = sym.Matrix(D_mat[0:m - 1, 0:m])  # reduces D_mat to have deltas in between vectors entries
        D_mat_red = sym.Matrix(- np.eye(m - 1) + np.eye(m - 1, k=1))
        D_mat_red = sym.Matrix(D_mat_red[0:m-2,0:m-1])  # rm last row in matrix
        oneone_vec = sym.ones(m, 1)
        oneone_vec_short = sym.ones(m - 1, 1)
        one_vec = np.zeros((m, 1)); one_vec[0] = 1

        ################################################################################################################
        # Symbolic scalars
        ################################################################################################################
        sym_sc = {'m_t': sym.symbols('m_t'),
                  'Pmax_kW': sym.symbols('Pmax_kW'),
                  'Fmax_kN': sym.symbols('Fmax_kN'),
                  'Fmin_kN': sym.symbols('Fmin_kN'),
                  'vmax_mps': sym.symbols('vmax_mps'),
                  'axmax_mps2': sym.symbols('axmax_mps2'),
                  'aymax_mps2': sym.symbols('aymax_mps2'),
                  'dF_kN_pos': sym.symbols('dF_kN_pos'),
                  'dF_kN_neg': sym.symbols('dF_kN_neg')
                  }
        vini_mps = sym.symbols('vini_mps')

        # --- Assign to class
        self.sym_sc = sym_sc
        self.vini_mps = vini_mps

        # --- Assign numeric values to non-changeable parameters
        sym_sc_ = {'m_t_': 1,
                   'Pmax_kW_': 215,
                   'Fmax_kN_': 7.1,
                   'Fmin_kN_': - 20,
                   'vmax_mps_': 70,
                   'axmax_mps2_': 10,
                   'aymax_mps2_': 10,
                   'dF_kN_pos_': 2,
                   'dF_kN_neg_': - 2
                  }
        self.sym_sc_ = sym_sc_

        ################################################################################################################
        # Symbolic states vector
        ################################################################################################################
        x = {}  # empty states dict
        p_kappa = {}  # empty parameters dict for kappa
        p_d_s = {}  # empty parameters dict for delta_s-values
        for i in range(0, m):
            v_sym_str = 'v' + str(i)
            k_sym_str = 'k' + str(i)
            ds_sym_str = 'ds' + str(i)
            x['v{0}'.format(i)] = sym.symbols(v_sym_str)
            p_kappa['k{0}'.format(i)] = sym.symbols(k_sym_str)
            if not i >= m - 1:  # m - 1 parameters available in vector delta_s
                p_d_s['ds{0}'.format(i)] = sym.symbols(ds_sym_str)

        # Only add slack variables if obj. function requires some
        if self.obj_func == 'slacks':
            for i in range(m, 2 * m):
                ev_sym_str = 'ev' + str(i - m)
                x['ev{0}'.format(i - m)] = sym.symbols(ev_sym_str)

        # --- Create vector containing all optimization variables
        v = sym.Matrix([[x for x in x.values()]])
        # --- Create vector containing all kappa parameters
        kappa = sym.Matrix([[p for p in p_kappa.values()]])
        # --- Create vector containing all delta_s parameters
        delta_s = sym.Matrix([[p for p in p_d_s.values()]])

        print('# --- Optimization vector v --- #')
        sym.pprint(v)
        print('# --- Parameter vector kappa --- #')
        sym.pprint(kappa)
        print('# --- Parameter vector delta_s --- #')
        sym.pprint(delta_s)

        ################################################################################################################
        # Symbolic objective
        ################################################################################################################
        # Extract velocity optimization variables depending on chosen objective function
        if self.obj_func == 'slacks':
            vel = sym.MatMul(sym.eye(m, 2 * m), v.T).doit().T
        else:
            vel = v
        vel_short = sym.eye(m - 1, m) * vel.T

        v_inv_short = vel_short.applyfunc(lambda x: 1 / x)
        print('# --- Elementwise inverse of v_short --- #')
        sym.pprint(v_inv_short)


        # --- Define the correct objective function
        J = 0
        if self.obj_func == 'slacks':
            # Extract slacks on velocity
            ev_m = np.eye(m, 2 * m, k=m)
            ev_vel = sym.MatMul(sym.Matrix(ev_m), v.T).doit().T

            # Objective including slacks
            J = v_inv_short.T * delta_s.T + ev_vel_w * ev_vel * sym.ones(m, 1)
        elif self.obj_func == 'objdv':
            # Objective using J = min. sum(v - vmax) ** 2
            J = (vel - sym.ones(1, m) * sym_sc_['vmax_mps_']).applyfunc(lambda x: sym.Pow(x, 2)) * sym.ones(m, 1)

        print('# --- Objective J --- #')
        sym.pprint(J)
        J_jac = J.jacobian(v)
        print('# --- Jacobian of J --- #')
        sym.pprint(J_jac)
        print('# --- Hessian of J --- #')
        J_hess = sym.hessian(J, v)
        sym.pprint(J_hess)

        ################################################################################################################
        # Symbolic constraints
        ################################################################################################################
        dv_mps = sym.MatMul(D_mat, vel.T).doit()
        inv_p_d_s_pm = delta_s.applyfunc(lambda x: 1 / x)

        # --- Create diagonal-matrix from short velocity-vector
        i = 0
        vel_short_diag = sym.eye(m - 1, m - 1)
        for e in vel_short:  # creates diagonal matrix
            vel_short_diag[i, i] = e; i += 1
        inv_dt_ps = sym.MatMul(vel_short_diag, inv_p_d_s_pm.T).doit()

        inv_dt_ps_diag = sym.eye(m - 1, m - 1)
        i = 0
        for e in inv_dt_ps.T:  # creates diagonal matrix
            inv_dt_ps_diag[i, i] = e
            i += 1
        a_mps2 = sym.MatMul(inv_dt_ps_diag, dv_mps).doit()

        # Calculate force
        F_kN = sym.MatMul(sym_sc['m_t'], a_mps2)

        # --- Velocity initial
        v_sym_cst = sym.MatMul(sym.Matrix(one_vec).T, vel.T).doit() - sym.MatMul(vini_mps, sym.ones(1, 1))
        # --- Velocity box
        if self.obj_func == 'slacks':
            v_sym_box_cst = vel.T - sym.MatMul(sym_sc['vmax_mps'], sym.ones(m, 1)) - ev_vel.T
        else:
            v_sym_box_cst = vel.T - sym.MatMul(sym_sc['vmax_mps'], sym.ones(m, 1))

        # Jacobian from velocity-initial-constraint
        v_sym_cst_jac = v_sym_cst.jacobian(v)

        # Jacobian from velocity-box-constraint
        v_sym_box_cst_jac = v_sym_box_cst.jacobian(v)

        # --- Force pos.
        F_sym_pos_cst = (F_kN - sym.MatMul(sym_sc['Fmax_kN'], sym.Matrix(oneone_vec_short))).doit()

        # --- Force neg.
        F_sym_neg_cst = (F_kN + sym.MatMul(sym_sc['Fmin_kN'], sym.Matrix(oneone_vec_short))).doit()

        # Jacobian from force-constraints
        F_sym_pos_cst_jac = F_sym_pos_cst.jacobian(v)
        F_sym_neg_cst_jac = F_sym_neg_cst.jacobian(v)

        # --- Delta-force box
        # F_kN = sym.MatMul(sym_sc['m_t'], a_mps2)  # rm last acceleration value
        dF_sym_pos_cst = sym.MatMul(D_mat_red, F_kN) -\
                     sym.MatMul(sym.ones(m - 2, 1), sym_sc['dF_kN_pos'])
        dF_sym_neg_cst = sym.MatMul(sym.ones(m - 2, 1), sym_sc['dF_kN_neg']) -\
                         sym.MatMul(D_mat_red, F_kN)


        # Jacobian from Delta-force box
        dF_sym_pos_cst_jac = dF_sym_pos_cst.jacobian(v)
        dF_sym_neg_cst_jac = dF_sym_neg_cst.jacobian(v)


        # --- Power
        #F_kN = sym.MatMul(sym_sc['m_t'], a_mps2)
        F_kN_diag = sym.eye(m - 1)
        for i in range(0, m - 1):  # creates diagonal matrix
            F_kN_diag[i, i] = F_kN[i, 0]

        P_sym_cst = sym.MatMul(F_kN_diag, vel_short) -\
                    sym.MatMul(sym_sc['Pmax_kW'], sym.Matrix(oneone_vec_short))

        # Jacobian of power constraint
        P_sym_cst_jac = P_sym_cst.jacobian(v)

        # --- Tire
        kappa_diag = sym.eye(m - 1)
        for i in range(0, m - 1):  # creates diagonal matrix
            kappa_diag[i, i] = kappa[i]
        v2 = vel.applyfunc(lambda x: x ** 2)
        v2 = sym.MatMul(sym.eye(m - 1, m), v2.T).doit()

        ax_norm = sym.MatMul(a_mps2, 1 / sym_sc_['axmax_mps2_']).doit()
        ay_norm = sym.MatMul(sym.MatMul(kappa_diag, v2), 1 / sym_sc_['aymax_mps2_']).doit()

        # Adding 4 tire constraints to implement abs()-function
        Tre_sym_cst1 = (ax_norm + ay_norm - oneone_vec_short).doit()
        Tre_sym_cst2 = (- ax_norm + ay_norm - oneone_vec_short).doit()
        Tre_sym_cst3 = (- ax_norm - ay_norm - oneone_vec_short).doit()
        Tre_sym_cst4 = (ax_norm - ay_norm - oneone_vec_short).doit()

        # Jacobian of tire constraint
        Tre_sym_cst1_jac = Tre_sym_cst1.jacobian(v)
        Tre_sym_cst2_jac = Tre_sym_cst2.jacobian(v)
        Tre_sym_cst3_jac = Tre_sym_cst3.jacobian(v)
        Tre_sym_cst4_jac = Tre_sym_cst4.jacobian(v)

        if self.obj_func == 'slacks':
            # --- Slack velocity constraint
            ev_vel_cst = ev_vel.doit()

            # Jacobian of slack velocity constraint
            ev_vel_cst_jac = ev_vel_cst.jacobian(v)

        # --- Assign state vector
        self.x = x
        self.kappa = kappa
        self.delta_s = delta_s

        # --- Assign Objective
        self.J_jac = J_jac
        self.J_hess = J_hess

        # --- Assign constraints
        self.F_sym_pos_cst = F_sym_pos_cst
        self.F_sym_pos_cst_jac = F_sym_pos_cst_jac
        self.F_sym_neg_cst = F_sym_neg_cst
        self.F_sym_neg_cst_jac = F_sym_neg_cst_jac
        self.v_sym_cst = v_sym_cst
        self.v_sym_cst_jac = v_sym_cst_jac
        self.v_sym_box_cst = v_sym_box_cst
        self.v_sym_box_cst_jac = v_sym_box_cst_jac
        self.P_sym_cst = P_sym_cst
        self.P_sym_cst_jac = P_sym_cst_jac
        self.Tre_sym_cst1 = Tre_sym_cst1
        self.Tre_sym_cst1_jac = Tre_sym_cst1_jac
        self.Tre_sym_cst2 = Tre_sym_cst2
        self.Tre_sym_cst2_jac = Tre_sym_cst2_jac
        self.Tre_sym_cst3 = Tre_sym_cst3
        self.Tre_sym_cst3_jac = Tre_sym_cst3_jac
        self.Tre_sym_cst4 = Tre_sym_cst4
        self.Tre_sym_cst4_jac = Tre_sym_cst4_jac
        if self.obj_func == 'slacks':
            self.ev_vel_cst = ev_vel_cst
            self.ev_vel_cst_jac = ev_vel_cst_jac
        self.dF_sym_pos_cst = dF_sym_pos_cst
        self.dF_sym_pos_cst_jac = dF_sym_pos_cst_jac
        self.dF_sym_neg_cst = dF_sym_neg_cst
        self.dF_sym_neg_cst_jac = dF_sym_neg_cst_jac


    def subs_symbolics(self):
        ################################################################################################################
        # Substituting symbolic expressions by numeric values
        ################################################################################################################
        x = self.x
        kappa = self.kappa
        delta_s = self.delta_s

        vini_mps = self.vini_mps

        ################################################################################################################
        # Replace hard parameter symbolics in constraints
        ################################################################################################################
        # --- Force constraint pos.
        self.F_sym_pos_cst = self.subs_syms(self.F_sym_pos_cst)
        self.F_sym_pos_cst_jac = self.subs_syms(self.F_sym_pos_cst_jac)
        # --- Force constraint neg.
        self.F_sym_neg_cst = self.subs_syms(self.F_sym_neg_cst)
        self.F_sym_neg_cst_jac = self.subs_syms(self.F_sym_neg_cst_jac)
        # --- Velocity box
        self.v_sym_box_cst = self.subs_syms(self.v_sym_box_cst)
        self.v_sym_box_cst_jac = self.subs_syms(self.v_sym_box_cst_jac)
        # --- Power
        self.P_sym_cst = self.subs_syms(self.P_sym_cst)
        self.P_sym_cst_jac = self.subs_syms(self.P_sym_cst_jac)
        # -- Tire
        self.Tre_sym_cst1 = self.subs_syms(self.Tre_sym_cst1)
        self.Tre_sym_cst1_jac = self.subs_syms(self.Tre_sym_cst1_jac)
        self.Tre_sym_cst2 = self.subs_syms(self.Tre_sym_cst2)
        self.Tre_sym_cst2_jac = self.subs_syms(self.Tre_sym_cst2_jac)
        self.Tre_sym_cst3 = self.subs_syms(self.Tre_sym_cst3)
        self.Tre_sym_cst3_jac = self.subs_syms(self.Tre_sym_cst3_jac)
        self.Tre_sym_cst4 = self.subs_syms(self.Tre_sym_cst4)
        self.Tre_sym_cst4_jac = self.subs_syms(self.Tre_sym_cst4_jac)

        if self.obj_func == 'slacks':
            # --- Slack velocity
            self.ev_vel_cst = self.subs_syms(self.ev_vel_cst)
            self.ev_vel_cst_jac = self.subs_syms(self.ev_vel_cst_jac)

        # --- Delta-force box
        self.dF_sym_pos_cst = self.subs_syms(self.dF_sym_pos_cst)
        self.dF_sym_pos_cst_jac = self.subs_syms(self.dF_sym_pos_cst_jac)
        self.dF_sym_neg_cst = self.subs_syms(self.dF_sym_neg_cst)
        self.dF_sym_neg_cst_jac = self.subs_syms(self.dF_sym_neg_cst_jac)

        ################################################################################################################
        # Create function handles from symbolic expressions with necessary free parameters
        ################################################################################################################

        # --- Lambdification of objective
        if self.obj_func == 'slacks':
            self.fJ_jac = sym.lambdify([list(x.values()), list(delta_s.values())], self.J_jac, 'numpy')
            self.fJ_Hess = sym.lambdify([list(x.values()), list(delta_s.values())], self.J_hess, 'numpy')
        else:
            self.fJ_jac = sym.lambdify([list(x.values())], self.J_jac, 'numpy')
            self.fJ_Hess = sym.lambdify([list(x.values())], self.J_hess, 'numpy')

        # --- Lambdification of constraints
        # Force pos.
        self.fF_pos_cst = sym.lambdify([list(x.values()), list(delta_s.values())], self.F_sym_pos_cst, 'numpy')
        self.fF_pos_cst_jac = sym.lambdify([list(x.values()), list(delta_s.values())], self.F_sym_pos_cst_jac, 'numpy')
        # Force neg.
        self.fF_neg_cst = sym.lambdify([list(x.values()), list(delta_s.values())], self.F_sym_neg_cst, 'numpy')
        self.fF_neg_cst_jac = sym.lambdify([list(x.values()), list(delta_s.values())], self.F_sym_neg_cst_jac, 'numpy')
        # Initial velocity
        self.fv_cst = sym.lambdify([list(x.values()), vini_mps], self.v_sym_cst, 'numpy')
        self.fv_cst_jac = sym.lambdify([list(x.values())], self.v_sym_cst_jac, 'numpy')
        # Velocity box
        self.fv_box_cst = sym.lambdify([list(x.values())], self.v_sym_box_cst, 'numpy')
        self.fv_box_cst_jac = sym.lambdify([list(x.values())], self.v_sym_box_cst_jac, 'numpy')
        # Power
        self.fP_cst = sym.lambdify([list(x.values()), list(delta_s.values())], self.P_sym_cst, 'numpy')
        self.fP_cst_jac = sym.lambdify([list(x.values()), list(delta_s.values())], self.P_sym_cst_jac, 'numpy')
        # Tire
        self.fTre_cst1 = sym.lambdify([list(x.values()), list(kappa.values()), list(delta_s.values())],
                                      self.Tre_sym_cst1, 'numpy')
        self.fTre_cst1_jac = sym.lambdify([list(x.values()), list(kappa.values()), list(delta_s.values())],
                                          self.Tre_sym_cst1_jac, 'numpy')
        self.fTre_cst2 = sym.lambdify([list(x.values()), list(kappa.values()), list(delta_s.values())],
                                      self.Tre_sym_cst2, 'numpy')
        self.fTre_cst2_jac = sym.lambdify([list(x.values()), list(kappa.values()), list(delta_s.values())],
                                          self.Tre_sym_cst2_jac, 'numpy')
        self.fTre_cst3 = sym.lambdify([list(x.values()), list(kappa.values()), list(delta_s.values())],
                                      self.Tre_sym_cst3, 'numpy')
        self.fTre_cst3_jac = sym.lambdify([list(x.values()), list(kappa.values()), list(delta_s.values())],
                                          self.Tre_sym_cst3_jac, 'numpy')
        self.fTre_cst4 = sym.lambdify([list(x.values()), list(kappa.values()), list(delta_s.values())],
                                      self.Tre_sym_cst4, 'numpy')
        self.fTre_cst4_jac = sym.lambdify([list(x.values()), list(kappa.values()), list(delta_s.values())],
                                          self.Tre_sym_cst4_jac, 'numpy')

        if self.obj_func == 'slacks':
            # Slack velocity
            self.fev_vel_cst = sym.lambdify([list(x.values())], self.ev_vel_cst, 'numpy')
            self.fev_vel_cst_jac = sym.lambdify([list(x.values())], self.ev_vel_cst_jac, 'numpy')

        # Delta-force box
        self.fdF_pos_cst = sym.lambdify([list(x.values()), list(delta_s.values())], self.dF_sym_pos_cst, 'numpy')
        self.fdF_pos_cst_jac = sym.lambdify([list(x.values()), list(delta_s.values())],
                                            self.dF_sym_pos_cst_jac, 'numpy')
        self.fdF_neg_cst = sym.lambdify([list(x.values()), list(delta_s.values())], self.dF_sym_neg_cst, 'numpy')
        self.fdF_neg_cst_jac = sym.lambdify([list(x.values()), list(delta_s.values())],
                                            self.dF_sym_neg_cst_jac, 'numpy')

    def subs_syms(self, arg):
        for sym_scalar in self.sym_sc.values():
            arg = arg.subs(sym_scalar, self.sym_sc_[sym_scalar.name + '_'])

        return arg


    ####################################################################################################################
    # OSQP-solver
    ####################################################################################################################
    def osqp_init(self, x0_, vini_mps_, kappa_, delta_s_, b_online_mode):
        # solves: 1/2 * x.T P x + q.T x; s.t. l <= Ax <= u
        sol_osqp = self.sol_osqp

        P, q, A, l, u = self.get_osqp_mat(x0_, vini_mps_, kappa_, delta_s_)

        if b_online_mode:
            sol_osqp_stgs = {'verbose': False,
                             'eps_abs': 1e-3,
                             'eps_rel': 1e-3}
        else:
            sol_osqp_stgs = {'verbose': True,
                             'eps_abs': 1e-5,
                             'eps_rel': 1e-5}

        sol_osqp.setup(P=P, q=q, A=A, l=l, u=u, **sol_osqp_stgs)


    def osqp_solve(self):
        sol = self.sol_osqp.solve()  # x; y; info: obj_val, setup_time, solve_time, run_time
        # sol.info.obj_val
        # sol.info.solve_time
        # sol.info.run_time

        print("Entire QP run-time [ms]: ", sol.info.run_time * 1000)
        return sol.x


    def osqp_update_online(self, x0_, vini_mps_, kappa_, delta_s_):
        ################################################################################################################
        # Update OSQP-problem matrices here
        ################################################################################################################

        P, q, A, l, u = self.get_osqp_mat(x0_, vini_mps_, kappa_, delta_s_)

        # --- Px, Ax is data from sparse matrices
        self.sol_osqp.update(Px=P.data, q=q, Ax=A.data, l=l, u=u)


    def get_osqp_mat(self, x0_, vini_mps_, kappa_, delta_s_):
        m = self.m
        conv_type = np.float64

        ################################################################################################################
        # Retrieve numeric values from lambda functions
        ################################################################################################################
        self.x0 = x0_

        if self.obj_func == 'slacks':
            self.J_jac_ = np.array(self.fJ_jac(x0_, delta_s_)).astype(conv_type)
            self.J_Hess_ = np.array(self.fJ_Hess(x0_, delta_s_)).astype(conv_type)
        else:
            self.J_jac_ = np.array(self.fJ_jac(x0_)).astype(conv_type)
            self.J_Hess_ = np.array(self.fJ_Hess(x0_)).astype(conv_type)

        self.F_pos_cst_ = np.array(self.fF_pos_cst(x0_, delta_s_)).astype(conv_type)
        self.F_pos_cst_jac_ = np.array(self.fF_pos_cst_jac(x0_, delta_s_)).astype(conv_type)
        self.F_neg_cst_ = np.array(self.fF_neg_cst(x0_, delta_s_)).astype(conv_type)
        self.F_neg_cst_jac_ = np.array(self.fF_neg_cst_jac(x0_, delta_s_)).astype(conv_type)
        self.v_cst_ = np.array(self.fv_cst(x0_, vini_mps_)).astype(conv_type)
        self.v_cst_jac_ = np.array(self.fv_cst_jac(x0_)).astype(conv_type)
        self.v_box_cst_ = np.array(self.fv_box_cst(x0_)).astype(conv_type)
        self.v_box_cst_jac_ = np.array(self.fv_box_cst_jac(x0_)).astype(conv_type)
        self.P_cst_ = np.array(self.fP_cst(x0_, delta_s_)).astype(conv_type)
        self.P_cst_jac_ = np.array(self.fP_cst_jac(x0_, delta_s_)).astype(conv_type)
        self.Tre_cst1_ = np.array(self.fTre_cst1(x0_, kappa_, delta_s_)).astype(conv_type)
        self.Tre_cst1_jac_ = np.array(self.fTre_cst1_jac(x0_, kappa_, delta_s_)).astype(conv_type)
        self.Tre_cst2_ = np.array(self.fTre_cst2(x0_, kappa_, delta_s_)).astype(conv_type)
        self.Tre_cst2_jac_ = np.array(self.fTre_cst2_jac(x0_, kappa_, delta_s_)).astype(conv_type)
        self.Tre_cst3_ = np.array(self.fTre_cst3(x0_, kappa_, delta_s_)).astype(conv_type)
        self.Tre_cst3_jac_ = np.array(self.fTre_cst3_jac(x0_, kappa_, delta_s_)).astype(conv_type)
        self.Tre_cst4_ = np.array(self.fTre_cst4(x0_, kappa_, delta_s_)).astype(conv_type)
        self.Tre_cst4_jac_ = np.array(self.fTre_cst4_jac(x0_, kappa_, delta_s_)).astype(conv_type)

        if self.obj_func == 'slacks':
            self.ev_vel_cst_ = np.array(self.fev_vel_cst(x0_)).astype(conv_type)
            self.ev_vel_cst_jac_ = np.array(self.fev_vel_cst_jac(x0_)).astype(conv_type)

        self.dF_pos_cst_ = np.array(self.fdF_pos_cst(x0_, delta_s_)).astype(conv_type)
        self.dF_pos_cst_jac_ = np.array(self.fdF_pos_cst_jac(x0_, delta_s_)).astype(conv_type)
        self.dF_neg_cst_ = np.array(self.fdF_neg_cst(x0_, delta_s_)).astype(conv_type)
        self.dF_neg_cst_jac_ = np.array(self.fdF_neg_cst_jac(x0_, delta_s_)).astype(conv_type)

        ################################################################################################################
        # P, sparse
        ################################################################################################################
        P = sparse.csc_matrix(self.J_Hess_)

        ################################################################################################################
        # q
        ################################################################################################################
        q = self.J_jac_.T

        ################################################################################################################
        # A, sparse
        ################################################################################################################
        A = self.v_cst_jac_
        A = np.append(A, self.F_pos_cst_jac_, axis=0)  # axis=0 for appending along rows
        A = np.append(A, self.F_neg_cst_jac_, axis=0)
        A = np.append(A, self.v_box_cst_jac_, axis=0)
        A = np.append(A, self.P_cst_jac_, axis=0)
        A = np.append(A, self.Tre_cst1_jac_, axis=0)
        A = np.append(A, self.Tre_cst2_jac_, axis=0)
        A = np.append(A, self.Tre_cst3_jac_, axis=0)
        A = np.append(A, self.Tre_cst4_jac_, axis=0)
        if self.obj_func == 'slacks':
            A = np.append(A, self.ev_vel_cst_jac_, axis=0)
        A = np.append(A, self.dF_pos_cst_jac_, axis=0)
        A = np.append(A, self.dF_neg_cst_jac_, axis=0)
        A = sparse.csc_matrix(A)  # define A as being sparse

        ################################################################################################################
        # l
        ################################################################################################################
        l = np.array(0)  # initial velocity
        l = np.append(l,
                      np.array([- np.inf] * (m - 1)))  # force pos.
        l = np.append(l,
                      np.array([- np.inf] * (m - 1)))  # force neg.
        l = np.append(l,
                      np.array([- self.sym_sc_['vmax_mps_'] + 0.1] * m))  #  velocity box
        l = np.append(l,
                      np.array([- np.inf] * (m - 1)))  # power
        l = np.append(l,
                      np.array([- np.inf] * (m - 1)))  # tire1
        l = np.append(l,
                      np.array([- np.inf] * (m - 1)))  # tire2
        l = np.append(l,
                      np.array([- np.inf] * (m - 1)))  # tire3
        l = np.append(l,
                      np.array([- np.inf] * (m - 1)))  # tire4
        if self.obj_func == 'slacks':
            l = np.append(l,
                          np.array([0] * m))  # slack velocity
        l = np.append(l,
                      np.array([- np.inf] * (m - 2)))  # Delta-force box pos.
        l = np.append(l,
                      np.array([- np.inf] * (m - 2)))  # Delta-force box neg.

        ################################################################################################################
        # u: - g(x0)
        ################################################################################################################
        u = - (self.v_cst_)  # initial velocity
        u = np.append(u,
                      np.array( - (self.F_pos_cst_).T))  # force pos.
        u = np.append(u,
                      np.array(- (self.F_neg_cst_).T))  # force neg.
        if self.obj_func == 'slacks':
            u = np.append(u,
                          np.array(- (self.v_box_cst_).T))  # velocity box slacked
        else:
            u = np.append(u,
                          np.array([np.inf] * m))  # velocity box unbounded
        u = np.append(u,
                      np.array(- (self.P_cst_).T))  # power
        u = np.append(u,
                      np.array(- (self.Tre_cst1_).T))  # tire1
        u = np.append(u,
                      np.array(- (self.Tre_cst2_).T))  # tire2
        u = np.append(u,
                      np.array(- (self.Tre_cst3_).T))  # tire3
        u = np.append(u,
                      np.array(- (self.Tre_cst4_).T))  # tire4
        if self.obj_func == 'slacks':
            u = np.append(u,
                          np.array([np.inf] * m))  # slack velocity
        u = np.append(u,
                      np.array(- (self.dF_pos_cst_).T))  # Delta-force box pos.
        u = np.append(u,
                      np.array(- (self.dF_neg_cst_).T))  # Delta-force box neg.

        return P, q, A, l, u


    def vis_sol(self, v_op_, kappa_, delta_s_):
        plt.figure()

        plt.subplot(5, 2, 1)
        plt.plot(v_op_[0:self.m])
        plt.ylabel('Velocity in mps')
        plt.plot(v_op_[self.m:2 * self.m])
        plt.ylabel('Velocity in mps')
        plt.legend(['v_mps', 'slack v_mps'])
        plt.subplot(5, 2, 5)
        plt.plot(self.fF_pos_cst(v_op_, delta_s_) + self.sym_sc_['Fmax_kN_'])
        plt.ylabel('Force in kN')
        plt.legend(['F_kN'])
        plt.subplot(5, 2, 7)
        plt.plot(self.fdF_pos_cst(v_op_, delta_s_) + self.sym_sc_['dF_kN_pos_'])
        plt.ylabel('Delta-Force in kN')
        plt.legend(['dF_kN'])
        plt.subplot(5, 2, 9)
        plt.plot(self.fP_cst(v_op_, delta_s_) + self.sym_sc_['Pmax_kW_'])
        plt.ylabel('Power in kW')
        plt.legend(['P_kW'])
        plt.xlabel('Way-points')

        plt.subplot(5, 2, 2)
        plt.plot(self.fTre_cst1(v_op_, kappa_, delta_s_) + 1)
        plt.subplot(5, 2, 4)
        plt.plot(self.fTre_cst2(v_op_, kappa_, delta_s_) + 1)
        plt.subplot(5, 2, 6)
        plt.plot(self.fTre_cst3(v_op_, kappa_, delta_s_) + 1)
        plt.subplot(5, 2, 8)
        plt.plot(self.fTre_cst4(v_op_, kappa_, delta_s_) + 1)
        plt.ylabel('Tire usage')
        plt.legend(['Tre'])
        plt.xlabel('Way-points')

        plt.show()


if __name__ == '__main__':
    pass