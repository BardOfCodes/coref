from wacky import CfgNode as CN


def MachineSpec(machine_type):
    if machine_type == "local":
        project_dir = "/home/aditya/projects/coref"
        data_dir = "/media/aditya/DATA/data/coref"
    elif machine_type == "ccv":
        project_dir = "/users/aganesh8/data/aganesh8/projects/coref"
        data_dir = "/users/aganesh8/data/aganesh8/data/coref"
    specs = CN()
    specs.data_dir = data_dir
    specs.project_dir = project_dir
    return specs
