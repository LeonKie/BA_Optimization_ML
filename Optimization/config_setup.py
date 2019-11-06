
import configparser
import os


config = configparser.ConfigParser()
config['Model_Parameter'] = {'MASS': '10000',   # Vehicle mass in kg
                     'length_front': '2.1',     # Distance from front axle to COG in m
                     'length_rear': '1.8',      # Distance from rear axle to COG in m
                     'Pmax' : '200',            # Maximum Power accelleration in kW
                     'Cx' : '1',               # Longitudinal tire stiffness.     
                     'Cy' : '1',                # Lateral tire stiffness.      
                     'CA' : '0.4'              # Air resistance coefficient.  

                     }
config['Trajectory'] = {'steps_on_traj': '100',
                    'regression_poly_deg': '10',# fitting the drawing data with a Ploynome of this Degree
                    }
config['Optimizer_Model_2D'] = {
                    'step_size': '0.1', 

                    'acceleration': '4g',
                    'steering_angle_min' : '-90',
                    'steering_angle_max' : '90'
                    }

config['Optimizer_global'] = {
                    'step_size': '0.1', 
                    'solver': 'Newton',
                    }


cwd = os.getcwd()
print(cwd)

with open(cwd+'/BA_Optimization_ML/Optimization'+'/setup.ini', 'w') as configfile:
    config.write(configfile)


