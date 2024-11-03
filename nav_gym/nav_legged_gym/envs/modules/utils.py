import torch

#--------------Local Navigation Module Utils-----------------#
def distance_compare(robot, params, env_ids, nodes, env):  # TODO: remove from here
    """
    add the node if it is at least {thre}m away to the global graph
    """
    diff = torch.abs(robot.all_graph_nodes_abs[env_ids, :, :] - nodes[env_ids, :].unsqueeze(1))
    diff_norm = torch.norm(diff, dim=2)
    good_ids = torch.min(diff_norm, dim=1).values > params["thre"]  # ids that have explored far enough
    return good_ids

#-----------------------------------------------------------