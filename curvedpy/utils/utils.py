import numpy as np

# # Get the impact parameter and vector
# def getImpactParam(self, loc_hit, dir_hit):
#     # We create a line extending the dir_hit vector
#     line = list(zip(*[loc_hit + dir_hit*l for l in range(20)]))
#     # This line is used to construct the impact_vector
#     impact_vector = loc_hit - loc_hit.dot(dir_hit)*dir_hit
#     # We save the length of the impact_vector. This is called the impact parameter in
#     # scattering problems
#     impact_par = np.linalg.norm(impact_vector)
#     # We normalize the impact vector. This way we get, together with dir_hit, an 
#     # othonormal basis
#     if impact_par != 0:
#         impact_vector_normed = impact_vector/impact_par # !!! Check this, gives errors sometimes
#     else:
#         impact_vector_normed = impact_vector

#     return impact_vector_normed, impact_par


# Get the impact parameter and vector
def getImpactParam(ray_origin, ray_direction):
    impact_vector = ray_origin - ray_origin.dot(ray_direction)*ray_direction
    # We save the length of the impact_vector. This is called the impact parameter in
    # scattering problems
    impact_par = np.linalg.norm(impact_vector)
    # We normalize the impact vector. This way we get, together with dir_hit, an 
    # othonormal basis
    if impact_par != 0:
        impact_vector_normed = impact_vector/impact_par # !!! Check this, gives errors sometimes
    else:
        impact_vector_normed = impact_vector

    return impact_vector_normed, impact_par