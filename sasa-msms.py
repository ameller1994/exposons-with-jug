import os
import json

import jug
import glob
import numpy as np

from libexposon.pipeline import featurize, cluster
from libexposon.tasks import msm2file


# with open(os.path.join(os.path.dirname(__file__),
#                        'proteins.json'), 'r') as f:
#     CONFIGS = json.load(f)

CONFIGS = {
    # "myh7-full-chimera": {
    #    'pid': 14480,
    #    'clusterings': [
    #         {
    #             'cluster_radii': ['7.34'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #         }
    #    ],
    # },
    # "myh7-ho-chimera": {
    #    'pid': 14481,
    #    'clusterings': [
    #         {
    #             'cluster_radii': ['7.34'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #         }
    #    ],
    # },
    # "myh7-lower-abi-chimera": {
    #    'pid': 14482,
    #    'clusterings': [
    #         {
    #             'cluster_radii': ['7.34'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #         }
    #    ],
    # },
    # "myh7-upper-chimera": {
    #    'pid': 14483,
    #    'clusterings': [
    #         {
    #             'cluster_radii': ['7.34'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #         }
    #    ],
    # },
    # "myh7": {
    #    'pid': 14485,
    #    'clusterings': [
    #         {
    #             'cluster_numbers': ['5000'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #         }
    #    ],
    # },
    # "myo10": {
    #    'pid': 14486,
    #    'clusterings': [
    #         {
    #             'cluster_numbers': ['5000'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #         }
    #    ],
    # },
    # "myo5a": {
    #    'pid': 14487,
    #    'clusterings': [
    #         {
    #             'cluster_numbers': ['5000'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #         }
    #    ],
    # },
    # "mut-myo10": {
    #    'pid': 14488,
    #    'clusterings': [
    #         {
    #             'cluster_numbers': ['5000'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #         }
    #    ],
    # },
    # "mut-myo5a": {
    #    'pid': 14489,
    #    'clusterings': [
    #         {
    #             'cluster_numbers': ['5000'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #         }
    #    ],
    # },
    # "mut_myh7_myo10_pocket": {
    #    'pid': 17420,
    #    'clusterings': [
    #         {
    #             'cluster_numbers': ['5000'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #         }
    #    ],
    # },
    # "mut_myh7_myo5a_pocket": {
    #    'pid': 17421,
    #    'clusterings': [
    #         {
    #             'cluster_numbers': ['5000'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #         }
    #    ],
    # },
    # 'myh7-s532p-5n69': {
    #   'pid': 17407,
    #   'clusterings': [
    #     {
    #       'cluster_radii': ['7.34'],
    #       'cluster_distance': 'euclidean',
    #       'cluster_algorithm': 'khybrid',
    #       'kmedoids_updates': 5
    #     }
    #   ]
    # },
    # 'myh7-r237w-5n69': {
    #   'pid': 17408,
    #   'clusterings': [
    #     {
    #       'cluster_radii': ['7.34'],
    #       'cluster_distance': 'euclidean',
    #       'cluster_algorithm': 'khybrid',
    #       'kmedoids_updates': 5
    #     }
    #   ]
    # },
    # 'myh7-i736t-5n69': {
    #   'pid': 17409,
    #   'clusterings': [
    #     {
    #       'cluster_radii': ['7.34'],
    #       'cluster_distance': 'euclidean',
    #       'cluster_algorithm': 'khybrid',
    #       'kmedoids_updates': 5
    #     }
    #   ]
    # },
    # 'myh7-v606m-5n69': {
    #   'pid': 17410,
    #   'clusterings': [
    #     {
    #       'cluster_radii': ['7.34'],
    #       'cluster_distance': 'euclidean',
    #       'cluster_algorithm': 'khybrid',
    #       'kmedoids_updates': 5
    #     }
    #   ]
    # },
    # 'myh7-r663h-5n69': {
    #   'pid': 17411,
    #   'clusterings': [
    #     {
    #       'cluster_radii': ['7.34'],
    #       'cluster_distance': 'euclidean',
    #       'cluster_algorithm': 'khybrid',
    #       'kmedoids_updates': 5
    #     }
    #   ]
    # },
    # 'myh7-m515t-5n69': {
    #   'pid': 17412,
    #   'clusterings': [
    #     {
    #       'cluster_radii': ['7.34'],
    #       'cluster_distance': 'euclidean',
    #       'cluster_algorithm': 'khybrid',
    #       'kmedoids_updates': 5
    #     }
    #   ]
    # },
    # 'myh7-r453c-5n69': {
    #   'pid': 17413,
    #   'clusterings': [
    #     {
    #       'cluster_radii': ['7.34'],
    #       'cluster_distance': 'euclidean',
    #       'cluster_algorithm': 'khybrid',
    #       'kmedoids_updates': 5
    #     }
    #   ]
    # },
    # 'myh7-r403q-5n69': {
    #   'pid': 17414,
    #   'clusterings': [
    #     {
    #       'cluster_radii': ['7.34'],
    #       'cluster_distance': 'euclidean',
    #       'cluster_algorithm': 'khybrid',
    #       'kmedoids_updates': 5
    #     }
    #   ]
    # },
    # 'myh7-wt-5n69': {
    #   'pid': 17415,
    #   'clusterings': [
    #     {
    #       'cluster_radii': ['7.34'],
    #       'cluster_distance': 'euclidean',
    #       'cluster_algorithm': 'khybrid',
    #       'kmedoids_updates': 5
    #     }
    #   ]
    # },
    # "myh11-gg-4pa0": {
    #    'pid': 17422,
    #    'clusterings': [
    #         {
    #             'cluster_numbers': ['5000'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #         }
    #    ],
    # },
    # "myh13-oc-4pa0": {
    #    'pid': 17423,
    #    'clusterings': [
    #         {
    #             'cluster_numbers': ['5000'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #         }
    #    ],
    # },
    # "myh7-wt-4pa0-omm": {
    #    'pid': 17424,
    #    'mask': '/project/bowmore/ameller/projects/jugbooks/cluster_input/myh7-wt-4pa0-omm-trajectory-mask.npy',
    #    'clusterings': [
    #         {
    #             'cluster_radii': [8.4],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #             'lag_times': '10:250:12',
    #             'timestep': 25
    #         }
    #    ],
    #    'lag_time': 100,
    #    'model_cluster_radius': 8.4,
    # },
    # "myh7-rat-4pa0-omm": {
    #    'pid': 17431,
    #    'mask': '/project/bowmore/ameller/projects/jugbooks/cluster_input/myh7-rat-4pa0-omm-trajectory-mask.npy',
    #    'clusterings': [
    #         {
    #             'cluster_radii': [8.4],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #             'lag_times': '10:250:12',
    #             'timestep': 25
    #         }
    #    ],
    #    'lag_time': 100,
    #    'model_cluster_radius': 8.4,
    # },
    # "myh7-chimera-4pa0-omm": {
    #    'pid': 17432,
    #    'clusterings': [
    #         {
    #             'cluster_radii': [8.4],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #             'lag_times': '10:250:12',
    #             'timestep': 25
    #         }
    #    ],
    #    'lag_time': 100,
    #    'model_cluster_radius': 8.4,
    # },
    # "hs-cardiac-5n6a": {
    #    'pid': 17433,
    #    'mask': '/project/bowmore/ameller/projects/jugbooks/cluster_input/hs-cardiac-5n6a-trajectory-mask.npy',
    #    'clusterings': [
    #         {
    #             'cluster_numbers': ['5000'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #             'lag_times': '10:250:12',
    #             'timestep': 25
    #         }
    #    ],
    #    # 'lag_time': 100,
    #    # 'model_cluster_radius': 8.4,
    # },
    # "gg-smooth-5n6a": {
    #    'pid': 17434,
    #    'mask': '/project/bowmore/ameller/projects/jugbooks/cluster_input/gg-smooth-5n6a-trajectory-mask.npy',
    #    'clusterings': [
    #         {
    #             'cluster_numbers': ['5000'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #             'lag_times': '10:250:12',
    #             'timestep': 25
    #         }
    #    ],
    #    # 'lag_time': 100,
    #    # 'model_cluster_radius': 8.4,
    # },
    # "oc-skeletal-5n6a": {
    #    'pid': 17435,
    #    'clusterings': [
    #         {
    #             'cluster_numbers': ['5000'],
    #             'cluster_distance': 'euclidean',
    #             'cluster_algorithm': 'khybrid',
    #             'kmedoids_updates': 5,
    #             'lag_times': '10:250:12',
    #             'timestep': 25
    #         }
    #    ],
    #    # 'lag_time': 100,
    #    # 'model_cluster_radius': 8.4,
    # },
    "myh7b-d515n": {
       'pid': 17428,
       'stride': 100,
       'clusterings': [
            {
                'cluster_radii': ['8.5'],
                'cluster_distance': 'euclidean',
                'cluster_algorithm': 'khybrid',
                'kmedoids_updates': 5,
                'lag_times': '1:20:1',
                'timestep': 1
            }
       ],
       # 'lag_time': 100,
       # 'model_cluster_radius': 8.4,
    },
}


for p, cfg in CONFIGS.items():
    cfg["trajectories"] = f"/project/bowmanlab/ameller/trajectories/processed/{p}/aligned-{cfg['pid']}/run*/clone*/*.xtc"
    cfg["topology"] = f"/project/bowmanlab/ameller/trajectories/topologies/{p}-prot-masses.pdb"

prior_count = 1

sc_sasa_filenames = {}
for protein, cfg in CONFIGS.items():
    trajectories = np.sort(
        [
            os.path.abspath(pathname)
            for pathname in glob.glob(cfg["trajectories"])])

    if 'mask' in cfg:
        mask = np.load(cfg['mask'])
        sc_sasa_filenames[protein] = featurize(
            tag=protein,
            trajectories=trajectories,
            topology=cfg['topology'],
            mask=mask,
            stride=cfg['stride']
        )
    else:
        sc_sasa_filenames[protein] = featurize(
            tag=protein,
            trajectories=trajectories,
            topology=cfg['topology'],
            stride=cfg['stride']
        )

cluster_results = {}
for protein, cfg in CONFIGS.items():
    print(protein)

    if 'mask' in cfg:
        mask = np.load(cfg['mask'])
        trajectories = np.sort(
            [
                os.path.abspath(pathname)
                for pathname in glob.glob(cfg["trajectories"])])[mask]
    else:
        trajectories = np.sort(
            [
                os.path.abspath(pathname)
                for pathname in glob.glob(cfg["trajectories"])])

    cluster_results[protein] = []

    for cluster_cfg in cfg['clusterings']:
        rslts = cluster(
            tag=protein,
            trajectories=trajectories,
            topology=cfg['topology'],
            sasa_sidechain_h5=sc_sasa_filenames[protein],
            **cluster_cfg,
            stride=cfg['stride']
        )
        cluster_results[protein].extend(rslts)



selected_cluster_results = {}
for protein, cluster_result_list in cluster_results.items():
    if "lag_time" not in CONFIGS[protein]:
        continue

    if 'cluster_radii' in CONFIGS[protein]['clusterings']:
        cluster_radii = [radius for cfg in CONFIGS[protein]['clusterings'] for radius in cfg['cluster_radii'] ]
        for radius, cluster_result in zip(cluster_radii,
                                          cluster_result_list):
            if CONFIGS[protein]["model_cluster_radius"] == radius:
                selected_cluster_results[protein] = cluster_result
                lag_time = CONFIGS[protein]["lag_time"]

                if cluster_result.assignments.can_load():
                    dirname, assigs_file = os.path.split(
                        jug.bvalue(cluster_result.assignments))

                    dirname = os.path.join(os.path.split(dirname)[0], 'models')
                    msm_filename = assigs_file.replace('-assignments.h5',
                                                       '-%02dprior-%slt-msm' %
                                                       (prior_count, lag_time))

                    assignments = cluster_result.assignments
                    msm2file(os.path.join(dirname, msm_filename),
                             assignments,
                             lag_time=lag_time)
