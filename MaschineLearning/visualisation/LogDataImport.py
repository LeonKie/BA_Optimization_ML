
import json


def get_data_from_line(file_path_in: str,
                       line_num: int):
    # skip header
    line_num = line_num + 1

    # extract a certain line number (based on time_stamp)
    with open(file_path_in) as file:
        # get to top of file (1st line)
        file.seek(0)

        # get header (":-1" in order to remove tailing newline character)
        file.readline()
        header = file.readline()[:-1]

        # extract line
        line = ""
        for _ in range(line_num):
            line = file.readline()

        # parse the data objects we want to retrieve from that line
        data = dict(zip(header.split(";"), line.split(";")))

        # decode
        # start_node;obstacle_pos_list;obstacle_radius_list;nodes_list;clip_pos
        start_node = json.loads(data['start_node'])
        obj_veh_data = json.loads(data['obj_veh'])
        obj_zone_data = json.loads(data['obj_zone'])
        nodes_list = json.loads(data['nodes_list'])
        clip_pos = json.loads(data['clip_pos'])

        s_list = json.loads(data['s_list'])
        pos_list = json.loads(data['pos_list'])
        vel_list = json.loads(data['vel_list'])
        a_list = json.loads(data['a_list'])
        psi_list = json.loads(data['psi_list'])
        kappa_list = json.loads(data['kappa_list'])

        # read action id and trajectory id (just in case we will reach the end of file
        action_id_prev = json.loads(data['action_id_prev'])
        traj_sel_idx = json.loads(data['traj_id_prev'])

        const_path_seg = np.array(json.loads(data['const_path_seg']))

        # get action selector
        line = file.readline()
        data = dict(zip(header.split(";"), line.split(";")))

        if not line == '':
            action_id = json.loads(data['action_id_prev'])
            traj_sel_idx = json.loads(data['traj_id_prev'])
        else:
            action_id = action_id_prev

    return (kappa_list['straight'][0],vel_list['straight'][0])


def get_data(file_path_in: str):
    # skip header
    DataOut=[]
    # extract a certain line number (based on time_stamp)
    with open(file_path_in) as file:
        # get to top of file (1st line)
        file.seek(0)
        # get header (":-1" in order to remove tailing newline character)
        file.readline()
        header = file.readline()[:-1]

        while True:
            # extract line
            line = file.readline()
            if not line:
                 break
            # parse the data objects we want to retrieve from that line
            data = dict(zip(header.split(";"), line.split(";")))

            
            # decode
            # start_node;obstacle_pos_list;obstacle_radius_list;nodes_list;clip_pos
            s_list = json.loads(data['s_list'])
            pos_list = json.loads(data['pos_list'])
            vel_list = json.loads(data['vel_list'])
            a_list = json.loads(data['a_list'])
            psi_list = json.loads(data['psi_list'])
            kappa_list = json.loads(data['kappa_list'])
            #Append Data
            DataOut.append((kappa_list['straight'][0],vel_list['straight'][0]))

    return DataOut

#file_path_data="/Users/leonkiesgen/Documents/Python/mod_local_trajectory/logs/ltpl/2019_12_01/15_00_59_data.csv"
#df=get_data(file_path_data)

#print(df[1800])