import numpy as np




def to_cam_nuv(cam_loc, look_at, up):
	# If the v is in the -z direction and up is taken in the +y
	# Then n points into the +x direction and u in the +y
	# It is a right handed system
	
	# But first we need to translate the objects so that the camera is in the
	# origin of the camera frame
	T = np.array([  [1.0, 0.0, 0.0, -cam_loc[0]], \
					[0.0, 1.0, 0.0, -cam_loc[1]], \
					[0.0, 0.0, 1.0, -cam_loc[2]], \
					[0.0, 0.0, 0.0,      1.0  ]  \
					])
	
	# Vector pointing into the look direction of the camera
	v = look_at-cam_loc
	v = v/np.linalg.norm(v)
	# Vector in the normal plane of v
	n = np.cross(v, up)
	n = n/np.linalg.norm(n)
	# Vector in the up direction and ortho to v and n
	u = np.cross(n, v)
	u = u/np.linalg.norm(u)

	# A 4x4 matrix containing the rotation to the camera frame
	# It rotates homogeneous vectors in the object frame to the camera frame
	# IF the camera is at 0,0,0 and points into the -z direction, the only
	# difference between object and camera coordinates is v = -z
	R_vnu = np.array([	\
						[*n, 			0.0], \
						[*u,   			0.0], \
						[*v,			0.0], \
						[0.0, 0.0, 0.0, 1.0]\
						])



	return R_vnu@T



# ## ## ## ## ## ## ## ## ## ## ## ## 
# Orthogonal NUV projection to clip coordinates
# ## ## ## ## ## ## ## ## ## ## ## ## 
def S_nuv(left_nuv, right_nuv, bottom_nuv, top_nuv, near_nuv, far_nuv):
	# All clippings (left, right, etc) are in camera coordinates!
	
	# Note that near_nuv and far_nuv are in camera coordinates and not 
	# like in Angel, in object coordinates!
	
	# First we need to center our view volume to the clipping canonical
	# location, which is centered at the origin
	# T = np.array([\
	# 	[1, 0, 0, -(left_nuv+right_nuv)/2.0],\
	# 	[0, 1, 0, -(top_nuv+bottom_nuv)/2.0],\
	# 	[0, 0, 1, -(far_nuv+near_nuv)/2.0],\
	# 	[0, 0, 0,         1.       ]\
	# 	])
	# We work in proper nuv coordinates. So compared to Angel, the 3,4 component
	# is not (far + near)/2 but -(far + near)/2

	# Then we need to scale the volume to be 2x2x2
	S = np.array([\
		[     near_nuv*2.0/(right_nuv-left_nuv),	0,       		0,  0],\
		[     0,                near_nuv*2.0/(top_nuv-bottom_nuv), 	0,	0],\
		[	  0,       			0,       							1,  0],\
		[     0,                0,              					0, 	1]\
		])
	# S = np.array([\
	# 	[     2.0/(right_nuv-left_nuv),	0,       			0,         				0],\
	# 	[     0,                2.0/(top_nuv-bottom_nuv), 	0,						0],\
	# 	[	  0,       			0,       					2./(far_nuv-near_nuv),  0],\
	# 	[     0,                0,              			0, 		        		1]\
	# 	])
	# Note: we use nuv coordinates, thus that scaling is 2./(far-near). In Angel
	# far and near are given in object coordinates and he uses 2./(near-far)!

	return S#@T



def N_perspective_nuv(near_nuv, far_nuv):

	a = (near_nuv+far_nuv)/(far_nuv-near_nuv)
	b =  -2*(near_nuv*far_nuv)/(far_nuv-near_nuv)

	N = np.array([\
		[1, 0, 0, 0],\
		[0, 1, 0, 0],\
		[0, 0, a, b],\
		[0, 0, 1, 0]\
		])

	return N

def perspective_division(vec):
	# if the projection plane is at z=1 in camera coordinates
	# we need to project like this:
	# x_p = x/z
	# y_p = y/z
	# z_p = z/z = 1
	# This transformation is obtained by dividing of a vector by 
	# its fourth component
	# Doing this means you loose information about the depth
	# of the vertex
	return vec/vec[3]


def pipeline(cam_loc, look_at, up, left_nuv, right_nuv, bottom_nuv, top_nuv, near_nuv, far_nuv):
	# Transform to camera coordinates
	C = to_cam_nuv(cam_loc, look_at, up)
	# Scale to clipping size
	S = S_nuv(left_nuv, right_nuv, bottom_nuv, top_nuv, near_nuv, far_nuv)
	# Do perspective projection
	N = N_perspective_nuv(near_nuv, far_nuv)
	# Dont forget to do perspective division on your final vector!

	return N@S@C

def mock_vertex_shader(vertex, BH_LOC, scaler, cam_loc, look_at, up, left_nuv, right_nuv, bottom_nuv, top_nuv, near_nuv, far_nuv):
	direction = vertex-BH_LOC
	distance = np.linalg.norm(direction)
	# direction[3]=0.
	# amount = np.linalg.norm(distance)
	# direction = np.array([vertex[0], vertex[1], 0, 0])

	T_BH = M_translate(scaler/distance, direction/distance) # This is a move towards implementing a BH
	print
	M = pipeline(cam_loc, look_at, up, left_nuv, right_nuv, bottom_nuv, top_nuv, near_nuv, far_nuv)

	return perspective_division(M@T_BH@vertex)

##################################################
# BASIC MATRICES AND TRANSFORMATIONS
##################################################

def M_shear(th, ph):
	# This shearing matrix translates x, y, z to the
	# points :
	# x_p = x + z*cot(th)
	# y_p = y + z*cot(ph)
	# z_p = 0
	return np.array([\
		[1, 0, np.cot(th), 0],\
		[0, 1, np.cot(ph), 0],\
		[0, 0,      0,     0],\
		[0, 0, 		0, 	   1]
		])


def rot_z(th):
	return np.array([\
				[np.cos(th), -np.sin(th), 0, 0],\
				[np.sin(th), np.cos(th), 0, 0],\
				[0,0,1,0],\
				[0,0,0,1]\
				])
def rot_y(th):
	return np.array([\
				[np.cos(th), 0, np.sin(th), 0],\
				[0,1,0,0],\
				[-np.sin(th), 0, np.cos(th), 0],\
				[0,0,0,1]\
				])

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
def M_translate(amount, direction):
	'''Returns the 4x4 translation matrix for a 3D translation'''
	d = amount*direction/np.linalg.norm(direction)
	return np.array([\
		[1, 0, 0, d[0] ],\
		[0, 1, 0, d[1] ],\
		[0, 0, 1, d[2] ],\
		[0, 0, 0,   1   ]\
		])

## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
def translate(p3, amount, direction, return_matrix=False):
	'''Returns the translation of point p3 by amount in direction'''
	r = M_translate(amount, direction)@np.array([*p3, 1])
	return r[0:3]


#def M_rotate()


# ## ## ## ## ## ## ## ## ## ## ## ## 
# Camera coordinate system: vnu
# ## ## ## ## ## ## ## ## ## ## ## ## 
# def to_cam_vnu(cam_loc, look_at, up):
# 	# If the v is in the -z direction and up is taken in the +y
# 	# Then n points into the +x direction and u in the +y
# 	# It is a right handed system

# 	# Vector pointing into the look direction of the camera
# 	v = look_at-cam_loc
# 	v = v/np.linalg.norm(v)
# 	# Vector in the normal plane of v
# 	n = np.cross(v, up)
# 	n = n/np.linalg.norm(n)
# 	# Vector in the up direction and ortho to v and n
# 	u = np.cross(n, v)
# 	u = u/np.linalg.norm(u)

# 	# A 4x4 matrix containing the rotation to the camera frame
# 	# It rotates homogeneous vectors in the object frame to the camera frame
# 	R_vnu = np.array([[*v,0.0], [*n, 0.0], [*u,0.0], [0.0, 0.0, 0.0, 1.0]])

# 	# But first we need to translate the objects so that the camera is in the
# 	# origin of the camera frame
# 	T = np.array([  [1.0, 0.0, 0.0, -cam_loc[0]], \
# 					[0.0, 1.0, 0.0, -cam_loc[1]], \
# 					[0.0, 0.0, 1.0, -cam_loc[2]], \
# 					[0.0, 0.0, 0.0,      1.0  ]  \
# 					])

# 	return R_vnu@T


# ## ## ## ## ## ## ## ## ## ## ## ## 
# Orthogonal projection to clip coordinates
# ## ## ## ## ## ## ## ## ## ## ## ## 
# def ortho(left, right, bottom, top, near, far):
# 	T = np.array([\
# 		[1, 0, 0, -(far+near)/2.0],\
# 		[0, 1, 0, -(left+right)/2.0],\
# 		[0, 0, 1, -(top+bottom)/2.0],\
# 		[0, 0, 0,         1.       ]\
# 		])

# 	S = np.array([\
# 		[2./(far-near),       0,                0,         0],\
# 		[     0,        2.0/(right-left),       0,         0],\
# 		[     0,                0,       2.0/(top-bottom), 0],\
# 		[     0,                0,              0,         1]\
# 		])

# 	return S@T

# def projection_matrix_no_clipping():
# 	# The one at index 4,3 can be seen as "storing" the z coordinate
# 	# into the 4th index of the vector. Division of a vector by its fourth 
# 	# component then gives the proper perspective.
# 	return np.array([\
# 		[1,0,0,0],\
# 		[0,1,0,0],\
# 		[0,0,1,0],\
# 		[0,0,1,0]
# 		])